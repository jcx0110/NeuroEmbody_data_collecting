import sys
import time

class Logger:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m' # No Color

    @staticmethod
    def _timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def info(cls, msg):
        print(f"{cls.BLUE}[INFO]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def success(cls, msg):
        print(f"{cls.GREEN}[OK]  {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def warn(cls, msg):
        print(f"{cls.YELLOW}[WARN]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def error(cls, msg):
        print(f"{cls.RED}[ERR] {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()