import time
from functools import wraps


class FuncRunInfo:
    """类方法的修饰器."""
    def __init__(self, logger):
        self.logger = logger

	def __call__(self, func):
        """call."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            cls_name = func.__qualname__
            func_name = func.__name__
            self.logger.info(f'Start {cls_name} {func_name}')
            ss = time.time()
            ret = func(*args, **kwargs)
            run_time = round(time.time() - ss, 2)
            self.logger.info(
            	f'End {cls_name} {func_name} RunTime: {run_time}s')
            return ret
        return wrapper


class Register(dict):
    """类方法注册器."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dict = {}

	def __setitem__(self, key, value):
        """set."""
        self.dict[key] = value

	def __getitem__(self, key):
        """get."""
        return self._dict[key]

	def __contains__(self, key):
        """contains."""
        return key in self._dict

	def __str__(self):
        """str."""
        return str(self._dict)

	def keys(self):
        """key."""
        return self._dict.keys()

	def values(self):
        """values."""
        return self._dict.values()

	def items(self):
        """items."""
        return self._dict.items()

	def register(self, target):
        """注册."""
        self._dict[target.__name__] = target
