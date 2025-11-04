#!/usr/bin/env python3

"""
CMPUT 455 assignment 3 testing script

usage: python a3test.py [-h] your_submission test

positional arguments:
  your_submission  Path to your submissions .py file
  test             Path to the public test text file

options:
  -h, --help       show this help message and exit

"""

import sys
assert sys.version.startswith("3.8"), "Use Python 3.8"

from datetime import datetime
import os
import socket
import tempfile

import argparse
import contextlib
import contextlib
import math
from operator import attrgetter, itemgetter
from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from copy import copy
from typing import AsyncIterator, Dict, FrozenSet, Generator, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union, Awaitable, runtime_checkable
from itertools import starmap, zip_longest
import asyncio
from asyncio.subprocess import Process, create_subprocess_exec, PIPE
# Needed for Python3.8, in later versions the existing global is the same
from asyncio import TimeoutError

DYNAMIC_TIMELIMIT_CMDS = ("policy_moves", "load_patterns", "position_evaluation", "move_evaluation")
         
T = TypeVar("T")
# Default maximum command execution time in seconds
DEFAULT_TIMEOUT = 1
DYNAMIC_TIMEOUT = 1

USE_RESOURCE_LIMITS = False

# Total number of timeouts before every test is treated as a timeout
MAX_TIMEOUTS = 20
STATUS_PATTERN = re.compile(r"^= .*")
WHITESPACE_PATTERN = re.compile(r"^\s*#|^\s*$")
TIMELIMIT_PATTERN = re.compile(r"^timelimit\s+(\d+)\s*$")

# Color codes
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

def color_print(*args, color, **kwargs):
    """Works exactly like print() but will use a terminal color provided keyword argument
    If printing to a TTY (an interactive terminal)"""
    file = kwargs.get("file", sys.stdout)
    if not file.isatty():
        return print(*args, **kwargs)
    print(color, end="", file=file)
    print(*args, **kwargs, flush=False)
    print(RESET, end="", file=file, flush=kwargs.get("flush", False))


async def crash_guarded(fut: Awaitable[T], process: Process, timeout: float) -> Tuple[bool, T]:
    """Returns (didSucceed: bool, result | "timeout" | "crash")
    """
    async def succeed():
        try:
            return True, await fut
        except TimeoutError:
            return False, "timeout"
        except Exception:
            return False, "crash"
    async def crash():
        await process.wait()
        return False, "crash"
    try:
        tasks = map(asyncio.create_task, (succeed(), crash()))
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)
        if len(done) == 0:
            return False, "timeout"
        for f in pending:
            f.cancel()
        return await done.pop()
    except TimeoutError:
        return False, "timeout"


def dc_copy_from(attr: str, dest: T, src: T) -> T:
    """Return a copy of dest where the value of dest.<attr> has been replaced with the value of src.<attr>
    T must be a dataclass"""
    dest = copy(dest)
    object.__setattr__(dest, attr, getattr(src, attr))
    return dest

class StudentProgram:
    __path: Path
    __process: Optional[Process] = None
    __time_start: float = float("inf")
    __timeout_secs: float = 0.0
    timelimit_secs: Optional[Union[float, int]] = DEFAULT_TIMEOUT
    num_timeouts: int
    use_limits: bool

    def timeleft(self) -> float:
        secs_elapsed = time.time() - self.__time_start
        secs_left = self.__timeout_secs - secs_elapsed
        return max(0.001, secs_left)

    def __init__(self, submission: Path, use_limits: bool):
        self.__path = submission
        self.num_timeouts = 0
        self.use_limits = use_limits

    def reset_timer(self, timeout_secs: float):
        self.__timeout_secs = timeout_secs
        self.__time_start = time.time()

    async def maybe_read_error_text(self) -> str:
        error_text = ""
        err = self.__process.stderr
        with contextlib.suppress(TimeoutError):
            while not err.at_eof():
                error_text += (await asyncio.wait_for(err.readline(), timeout=0.01)).decode("utf-8")
        return error_text
    
    async def write_input(self, data: str):
        self.__process.stdin.write(bytes(data, encoding="utf-8"))
        await asyncio.wait_for(self.__process.stdin.drain(), self.timeleft())

    async def read_output(self) -> AsyncIterator[str]:
        out = self.__process.stdout
        while not out.at_eof():
            yield (await asyncio.wait_for(out.readline(), self.timeleft())).decode("utf-8")

    async def kill(self):
        if self.__process is None:
            return
        try:
            self.__process.kill()
            self.__process.stdin.write_eof()
            await asyncio.wait_for(self.__process.stdin.drain(), self.timeleft())
            self.__process.stdin.close()
            await asyncio.wait_for(self.__process.stdin.wait_closed(), self.timeleft())
        except TimeoutError:
            print("Timed out while trying to kill program", file=sys.stderr)
        except (ProcessLookupError, ConnectionResetError):
            pass
        finally:
            self.__process = None

    async def __assert_running(self):
        if self.__process is not None:
            return
        self.__process = await self.__load_program()
    
    def inject_resource_limit_code(self, python_prog: Path) -> Path:
        code = """
import resource
# No forking allowed
# (Don't need to worry about threads due to the GIL)
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
# One gigabyte
resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
"""
        file = tempfile.NamedTemporaryFile("wt", encoding="utf-8", suffix=".py", delete=False)
        file.write(code)
        file.write(python_prog.read_text())
        return Path(file.name)

    async def run_command(self, cmd: str) -> "Test":
        await self.__assert_running()
        result = await Test.from_student_program(cmd, self)
        if type(result) is TestTimeout:
            self.num_timeouts += 1
            print("Timed out", file=sys.stderr)
        return result

    @contextlib.contextmanager
    def temporary_time_limit(self, limit: Optional[Union[int, float]]):
        old = self.timeout_secs
        self.timeout_secs = limit
        yield
        self.timeout_secs = old

    async def run_test(self, test: "Test") -> "Test":
        if isinstance(test, TestTimelimit):
            self.timelimit_secs = test.limit
        if self.num_timeouts >= MAX_TIMEOUTS:
            return TestTimeout(test, 0.0, f"MAX_TIMEOUTS={MAX_TIMEOUTS} exceeded")
        if self.timelimit_secs is None:
            return await self.run_command(test.command)
        timeout_secs = self.timelimit_secs
        if test.command not in DYNAMIC_TIMELIMIT_CMDS:
            timeout_secs = DEFAULT_TIMEOUT
        # Always a small cushion for IO/scheduling jitter
        timeout_secs += 0.25
        await self.__assert_running()

        run_fut = self.run_command(test.command)

        self.reset_timer(timeout_secs)
        ok, result = await crash_guarded(run_fut, self.__process, timeout_secs)
        if ok:
            return result
        error_text = await self.maybe_read_error_text()
        await self.kill()
        if result == "timeout":
            self.num_timeouts += 1
            return TestTimeout(test, timeout_secs, error_text)
        elif result == "crash":
            return TestCrash(test, error_text)
        else:
            Exception(f"Unknown value for result: {result}")
    
    async def time_run_test(self, test: "Test") -> "Test":
        t0 = time.time()
        return (await self.run_test(test)).with_time_taken(time.time() - t0)

    async def __load_program(self) -> Process:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            injected_file = self.inject_resource_limit_code(self.__path) if self.use_limits else self.__path
            # Flag -u ensures unbuffered stdio
            return await create_subprocess_exec("python3", "-u", str(injected_file), stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env)
        except Exception as e:
            print(e)
            print(f"Failed to run `python3 '{self.__path}'`")
            sys.exit(1)


@dataclass(frozen=True)
class Invocation:
    submission: Path
    test: Path

    @staticmethod
    def from_args() -> "Invocation":
        parser = argparse.ArgumentParser(prog=f"python {sys.argv[0]}")
        parser.add_argument("your_submission", help="Path to your submissions .py file")
        parser.add_argument("test", help="Path to the public test text file")
        args = parser.parse_args()
        inv = Invocation(Path(args.your_submission), Path(args.test))
        assert inv.submission.exists(), f"invalid file path '{args.your_submission}'"
        assert inv.test.exists(), f"invalid file path '{args.test}'"
        return inv


async def chain_async(*a: List[Union[Iterator[T], AsyncIterator[T]]]) -> AsyncIterator[T]:
    "Works like itertools.chain, but allows a mix of async and non-async iterators"
    for i in a:
        if not isinstance(i, AsyncIterator):
            for ii in i:
                yield ii
        else:
            async for ii in i:
                yield ii

class TestParser:
    class IncompleteTestParse(Exception):
        pass
    class __CompleteTestParse(Exception):
        pass
    __parser: Generator[None, str, "Test"]
    marking: bool # Whether this is parsing a test for marking
    __result: Optional[Union["Test", Exception]] = None

    def __init__(self, marking):
        self.__parser = self.__create_parser()
        self.marking = marking
        next(self.__parser)
        
    def get_result(self):
        self.__parser.close()
        if isinstance(self.__result, Test):
            return self.__result
        if isinstance(self.__result, Exception):
            raise self.__result
        assert self.__result is None
        raise TestParser.IncompleteTestParse("Unknown error")
    
    def parse_lines(self, lines: Iterator[str]) -> "Test":
        "Simple parser interface for non async usage"
        with contextlib.suppress(TestParser.__CompleteTestParse):
            for line in lines:
                self.__feed_line(line)
        return self.get_result()
    
    async def parse_async_lines(self, lines: AsyncIterator[str]) -> "Test":
        with contextlib.suppress(TestParser.__CompleteTestParse):
            async for line in lines:
                self.__feed_line(line)
        return self.get_result()

    def __feed_line(self, line: str):
        if WHITESPACE_PATTERN.match(line):
            return
        try:
            self.__parser.send(line.strip())
        except StopIteration as e:
            self.__set_result(e.value)
            raise TestParser.__CompleteTestParse()

    def __set_result(self, e: Union["Test", Exception]):
        if self.__result is None:
            self.__result = e

    def __create_parser(self) -> Generator[None, str, "Test"]:
        counts_for_marks, command = self.__parse_command((yield), self.marking)
        result, status = yield from self.__parse_command_body()
        shared_args = (command, status, counts_for_marks, "")

        match = TIMELIMIT_PATTERN.match(command)
        if match is not None:
            with contextlib.suppress(Exception):
                dynamic_timeout = int(match.group(1))
                return TestTimelimit(*shared_args, dynamic_timeout)
        elif command == "score":
            with contextlib.suppress(Exception):
                a, b = map(float, result[0].split())
                return TestScore(*shared_args, a, b)

        if self.marking and len(result) and result[0].startswith("@"):
            return TestPattern(*shared_args, re.compile("\n".join(result)[1:], re.DOTALL))
        return TestLines(*shared_args, result)
    
    @staticmethod
    def __parse_command(line: str, marking: bool) -> Tuple[bool, str]:
        counts_for_marks = False
        if not marking:
            return counts_for_marks, line
        if line.startswith("?"):
            line = line[1:]
            counts_for_marks = True
        return counts_for_marks, line
    

    def __parse_command_body(self) -> Generator[None, str, Tuple[Tuple[str, ...], str]]:
        """Generator that accepts strings until a status line is found
        Returns a tuple of lines and the status line inside a tuple"""
        result = []
        try:
            while True:
                line = (yield)
                if STATUS_PATTERN.match(line):
                    status = line
                    break
                result.append(line)
        except GeneratorExit:
            self.__set_result(TestParser.IncompleteTestParse("Did not find status line"))
        else:
            return tuple(result), status


@dataclass(frozen=True)
class Test:
    command: str
    status: str
    # Following fields excluded from __eq__ and __hash__
    # To make caching test results easier
    counts_for_marks: bool = field(hash=False, compare=False)
    error_output: str = field(hash=False, compare=False)
    time_taken: float = field(init=False, hash=False, compare=False, default=-1.0)

   
    @staticmethod
    def from_test_file(test: Path) -> Tuple["Test", ...]:
        lines = iter(test.read_text().split("\n"))
        def consume():
            try:
                while True:
                    parser = TestParser(marking=True)
                    yield parser.parse_lines(lines)
            except (StopIteration, TestParser.IncompleteTestParse):
                pass
        return tuple(consume())

    @staticmethod
    async def from_student_program(command: str, process: StudentProgram) -> "Test":
        await process.write_input(f"{command}\n")
        lines = process.read_output()
        parser = TestParser(marking=False)
        test = await parser.parse_async_lines(chain_async((command,), lines))
        error_text = await process.maybe_read_error_text()
        object.__setattr__(test, "error_output", error_text)
        return test
    
    def with_time_taken(self, time_taken: float) -> "Test":
        clone = copy(self)
        object.__setattr__(clone, "time_taken", time_taken)
        return clone

@runtime_checkable
class TestWithResult(Protocol):
    result: str

@dataclass(frozen=True)
class TestLines(Test):
    result: Tuple[str, ...]

@dataclass(frozen=True)
class TestTimelimit(Test):
    limit: int
    result: str = field(init=False, default="")

@dataclass(frozen=True)
class TestScore(Test):
    a: float
    b: float

    @property
    def result(self) -> Tuple[str, ...]:
        return (f"{self.a} {self.b}",)

@dataclass(frozen=True)
class TestPattern(Test):
    pattern: re.Pattern

@dataclass(frozen=True)
class TestTimeout(Test):
    timeout: Union[int, float]

    def __init__(self, test: Test, timeout: Union[int, float], error_output: str):
        for field in Test.__annotations__:
            object.__setattr__(self, field, getattr(test, field))
        object.__setattr__(self, "status", "")
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "error_output", error_output)

@dataclass(frozen=True)
class TestCrash(Test):
    def __init__(self, test: Test, error_output: str):
        for field in Test.__annotations__:
            object.__setattr__(self, field, getattr(test, field))
        object.__setattr__(self, "status", "")
        object.__setattr__(self, "error_output", error_output)

@dataclass(frozen=True)
class TestResult:
    answer_key: Test
    student: Test
    status_matches: bool
    output_matches: bool
    time_out: bool
    crash: bool

    @property
    def counts_for_marks(self) -> bool:
        return self.answer_key.counts_for_marks

    @property
    def status_and_output_matches(self) -> bool:
        return self.status_matches and self.output_matches
    
    @property
    def marks_lost(self) -> bool:
        return self.counts_for_marks and not self.status_and_output_matches
    
    @staticmethod
    def better_test(a: "TestResult", b: "TestResult") -> "TestResult":
        if a.status_and_output_matches:
            return a
        return b

    @staticmethod
    def from_comparison(answer_key: Test, student: Test) -> "TestResult":
        if isinstance(student, TestTimeout):
            return TestResult(answer_key, student, False, False, True, False)
        if isinstance(student, TestCrash):
            return TestResult(answer_key, student, False, False, False, True)
        status_matches = answer_key.status == student.status
        assert isinstance(student, TestWithResult)
        if type(answer_key) is TestPattern:
            output_matches = bool(answer_key.pattern.match("\n".join(student.result)))
        else:
            assert isinstance(answer_key, TestWithResult)
            output_matches = answer_key.result == student.result
        return TestResult(answer_key, student, status_matches, output_matches, False, False)
    
    @staticmethod
    def from_comparisons(answer_key: Sequence[Test], student: Sequence[Test]) -> Tuple["TestResult", ...]:
        return tuple(starmap(TestResult.from_comparison, zip(answer_key, student)))

    def print_verbose(self, file=sys.stdout):
        print(f"Command: {self.answer_key.command}", file=file)
        if type(self.student) is TestTimeout:
            color_print(f"Program timed out", color=RED, file=file)
        elif type(self.student) is TestCrash:
            color_print(f"Program crashed", color=RED, file=file)
        elif self.output_matches:
            color_print("Output from command matches expected output", color=GREEN, file=file)
        elif type(self.answer_key) is TestPattern:
            print("Expected output matching the following regular expression:", file=file)
            color_print(self.answer_key.pattern.pattern, color=GREEN, file=file)
            print("Received:", file=file)
            color_print(*self.student.result, sep="\n", color=RED, file=file)
        elif isinstance(self.answer_key, TestWithResult):
            print("Expected:", file=file)
            color_print(*self.answer_key.result, sep="\n", color=GREEN, file=file)
            print("Received:", file=file)
            answer_key = "\n".join(self.answer_key.result)
            student = "\n".join(self.student.result)
            print_colored_diff(answer_key, student, file=file)
        else:
            raise Exception()

        print(file=file)

        if isinstance(self.student, (TestTimeout, TestCrash)):
            pass
        elif self.status_matches:
            color_print("Resulting status code is correct", color=GREEN, file=file)
        else:
            print("Expected status code:", file=file)
            color_print(self.answer_key.status, color=GREEN, file=file)
            print("Received status code:", file=file)
            print_colored_diff(self.answer_key.status, self.student.status, file=file)

        print(file=file)

        if self.student.error_output:
            print("Program outputted the following error text:", file=file)
            color_print(self.student.error_output, color=RED, file=file)

        if self.answer_key.counts_for_marks:
            print("This test will be marked.", file=file)
        else:
            print("This test will NOT be marked.", file=file)
    
def print_colored_diff(correct: str, incorrect: str, file=sys.stdout):
    for corr, inc in zip_longest(correct, incorrect):
        if inc is None:
            break
        color = GREEN if corr == inc else RED
        color_print(inc, color=color, sep="", end="", file=file)
    print(file=file)

def print_detailed_results(results: Sequence[TestResult]):
    for i, result in enumerate(results):
        if result.status_and_output_matches:
            continue
        print(f"=== Test {i} ===")
        result.print_verbose()
            

@dataclass(frozen=True)
class TestStatistics:
    test_count: int
    status_matches: int
    output_matches: int
    status_and_output_matches: int
    time_outs: int
    crashes: int
    total_time_taken: float

    @staticmethod
    def from_test_results(results: Sequence[TestResult]):
        return TestStatistics(
            len(results),
            *(sum(1 for result in results if getattr(result, attr))
                for attr in ("status_matches", "output_matches", "status_and_output_matches", "time_out", "crash")),
            sum(result.student.time_taken for result in results)
        )

    def fraction(self, attr: str) -> str:
        return f"{getattr(self, attr)} / {self.test_count}"

    def color(self, attr: str) -> str:
        return GREEN if getattr(self, attr) == self.test_count else RED
    
    def color_inv(self, attr: str) -> str:
        return RED if getattr(self, attr) == self.test_count else GREEN
    
    def summarize(self):
        color_print("Summary report:", color=BLUE)
        print(f"{self.test_count} tests performed")
        color_print(f"{self.fraction('status_matches')} output statuses matched.", color=self.color("status_matches"))
        color_print(f"{self.fraction('output_matches')} command outputs matched.", color=self.color("output_matches"))
        color_print(f"{self.fraction('time_outs')} tests timed out.", color=self.color_inv("time_outs"))
        color_print(f"{self.fraction('crashes')} tests crashed.", color=self.color_inv("crashes"))

    def public_marks(self):
        if self.test_count == 0:
            color_print("Nothing to mark", color=BLUE)
            return
        color_print("Marks report", color=BLUE)
        mark = round(math.floor(self.status_and_output_matches / self.test_count * 100) / 10, 1)
        if mark == 0 and self.status_and_output_matches != 0:
            mark = 0.1
        print(f"{self.status_and_output_matches} / {self.test_count} marked tests")

@dataclass(frozen=True)
class FullTestRun:
    results_all: Tuple[TestResult, ...]
    stats_all: TestStatistics
    results_marked: Tuple[TestResult, ...]
    stats_marked: TestStatistics
    machine_names: FrozenSet[str]
    date: datetime

    @staticmethod
    def from_comparisons(answer_key: Tuple[Test, ...], stu_tests: Tuple[Test, ...]) -> "FullTestRun":
        results_all = TestResult.from_comparisons(answer_key, stu_tests)
        stats_all = TestStatistics.from_test_results(results_all)
        
        for_marks = tuple((key, stu) for key, stu in zip(answer_key, stu_tests) if key.counts_for_marks)
        answer_key_for_marks = tuple(map(itemgetter(0), for_marks))
        student_for_marks = tuple(map(itemgetter(1), for_marks))
        results_for_marks = TestResult.from_comparisons(answer_key_for_marks, student_for_marks)
        stats_for_marks = TestStatistics.from_test_results(results_for_marks)

        machine_names = frozenset((socket.gethostname(),))
        date = datetime.now()
        return FullTestRun(results_all, stats_all, results_for_marks, stats_for_marks, machine_names, date)
    
    @staticmethod
    async def from_student_submission(submission: Path, test: Path, use_limits: bool) -> "FullTestRun":
        answer_key = Test.from_test_file(test)
        program = StudentProgram(submission, use_limits)
        stu_tests = tuple([await program.time_run_test(test) for test in answer_key])
        await program.kill()    
        return FullTestRun.from_comparisons(answer_key, stu_tests)
    
    @staticmethod
    def join(a: "FullTestRun", b: "FullTestRun") -> "FullTestRun":
        results_all = a.results_all + b.results_all
        results_marked = a.results_marked + b.results_marked
        stats_all = TestStatistics.from_test_results(results_all)        
        stats_for_marks = TestStatistics.from_test_results(results_marked)

        date = datetime.now()
        machines = a.machine_names | b.machine_names
        return FullTestRun(results_all, stats_all, results_marked, stats_for_marks, machines, date)
    
    @staticmethod
    def fold_best_tests(a: "FullTestRun", b: "FullTestRun") -> "FullTestRun":
        best = tuple(TestResult.better_test(aa, bb) for aa, bb in zip(a.results_all, b.results_all))
        best_ak = tuple(result.answer_key for result in best) 
        best_student = tuple(result.student for result in best)
        return FullTestRun.from_comparisons(best_ak, best_student)
        
test_submission = FullTestRun.from_student_submission
    
def reuse_test_result(test: Path, reuse: Dict[Path, FullTestRun]) -> Optional[Tuple[Path, FullTestRun]]:
    """Given a path to a test file and a dict of test runs, will look at the dict to see if the test has already been run
    Critically, this comparison will ignore the 'counts_for_marks' attribute

    If a match is found, returns the Path to the old test file and its FullTestRun with the 'counts_for_marks' and statistics updated according to the 'test' arg
    """
    answer_key = Test.from_test_file(test)
    for path, result in reuse.items():
        answer_key_reuse = tuple(map(attrgetter("answer_key"), result.results_all))
        if answer_key == answer_key_reuse:
            stu_tests_reuse = tuple(map(attrgetter("student"), result.results_all))
            stu_tests = tuple(
                dc_copy_from("counts_for_marks", stu, ak)
                for stu, ak in zip(stu_tests_reuse, answer_key)
            )
            new_result = FullTestRun.from_comparisons(answer_key, stu_tests)
            return path, new_result

def async_loop() -> asyncio.AbstractEventLoop:
    # https://stackoverflow.com/a/34114767
    loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

async def main():
    t0 = time.time()
    invocation = Invocation.from_args()
    run_result = await test_submission(invocation.submission, invocation.test, USE_RESOURCE_LIMITS)
    print_detailed_results(run_result.results_all)
    run_result.stats_all.summarize()
    run_result.stats_marked.public_marks()

    print("\nFinished after", round(time.time() - t0, 2), "seconds.")
    

if __name__ == "__main__" and not sys.flags.interactive:
    loop = async_loop()
    loop.run_until_complete(main())
