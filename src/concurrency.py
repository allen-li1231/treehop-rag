"""
Concurrency utilities for bounded parallel task execution.

Provides memory-efficient parallel processing by limiting the number of
concurrent pending tasks, preventing memory exhaustion with large datasets.
"""

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    wait,
    FIRST_COMPLETED,
)
from multiprocessing.context import BaseContext
from typing import Callable, Iterable, TypeVar, Generic, Any
from dataclasses import dataclass
from tqdm.auto import tqdm

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TaskResult(Generic[T, R]):
    """Container for a task result with its original identifier."""
    index: T
    result: R
    is_success: bool = True


def bounded_thread_pool_map(
    fn: Callable[..., R],
    items: Iterable[tuple[T, tuple[Any, ...]]],
    *,
    num_workers: int,
    max_pending: int,
    total: int | None = None,
    progress_bar: bool = True,
    progress_desc: str | None = None,
    progress_position: int | None = None,
    progress_leave: bool = True,
) -> list[TaskResult[T, R]]:
    """
    Execute tasks in a thread pool with bounded pending task count.
    
    This function limits the number of simultaneously pending (submitted but
    not completed) tasks to prevent memory exhaustion when processing large
    datasets.
    
    Special case: If num_workers <= 1, processes tasks sequentially in the
    main thread without spawning any worker threads.
    
    Args:
        fn: The function to execute for each item.
        items: An iterable of (index, args_tuple) pairs. Each args_tuple
            contains the arguments to pass to fn.
        num_workers: Number of worker threads. If <= 1, uses sequential processing.
        max_pending: Maximum number of tasks to keep pending at once.
            Controls memory usage. Ignored if num_workers <= 1.
        total: Total number of items for progress bar. If None, tries len(items).
        progress_bar: Whether to show progress bar. Default: True.
        progress_desc: Description for progress bar. Default: None.
        progress_position: Position for nested progress bars. Default: None.
        progress_leave: Whether to leave progress bar after completion. Default: True.
    
    Returns:
        List of TaskResult objects containing the index, result, and success status.
    
    Example:
        >>> def process(x, y):
        ...     return x + y
        >>> items = [(i, (i, i*2)) for i in range(1000)]
        >>> results = bounded_thread_pool_map(
        ...     process, items, num_workers=8, max_pending=50
        ... )
        
        # With nested progress bars:
        >>> for epoch in tqdm(range(10), desc="Epochs", position=0):
        ...     results = bounded_thread_pool_map(
        ...         process, items, num_workers=8, max_pending=50,
        ...         progress_desc="Processing", progress_position=1, progress_leave=False
        ...     )
    """
    # Determine total for progress bar
    if total is None:
        try:
            total = len(items)  # type: ignore
        except TypeError:
            total = None
    
    # Special case: sequential processing for num_workers <= 1
    if num_workers <= 1:
        results: list[TaskResult[T, R]] = []
        with tqdm(
            total=total,
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            disable=not progress_bar,
        ) as pbar:
            for index, args in items:
                try:
                    result = fn(*args)
                    task_result = TaskResult(index=index, result=result, is_success=True)
                except Exception as e:
                    task_result = TaskResult(index=index, result=e, is_success=False)
                
                results.append(task_result)
                pbar.update(1)
        
        return results
    
    # Parallel processing with thread pool
    items_iter = iter(items)
    results: list[TaskResult[T, R]] = []
    
    with tqdm(
        total=total,
        desc=progress_desc,
        position=progress_position,
        leave=progress_leave,
        disable=not progress_bar,
    ) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending: dict[Future, T] = {}
            
            # Submit initial batch of tasks
            for _ in range(max_pending):
                try:
                    index, args = next(items_iter)
                    future = executor.submit(fn, *args)
                    pending[future] = index
                except StopIteration:
                    break
            
            # Process results and submit new tasks as old ones complete
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                
                for future in done:
                    idx = pending.pop(future)
                    try:
                        result = future.result()
                        task_result = TaskResult(index=idx, result=result, is_success=True)
                    except Exception as e:
                        task_result = TaskResult(index=idx, result=e, is_success=False)
                    
                    results.append(task_result)
                    pbar.update(1)
                    
                    # Submit a new task if there are more items
                    try:
                        index, args = next(items_iter)
                        new_future = executor.submit(fn, *args)
                        pending[new_future] = index
                    except StopIteration:
                        pass
    
    return results


def bounded_process_pool_map(
    fn: Callable[..., R],
    items: Iterable[tuple[T, tuple[Any, ...]]],
    *,
    num_workers: int,
    max_pending: int,
    total: int | None = None,
    progress_bar: bool = True,
    progress_desc: str | None = None,
    progress_position: int | None = None,
    progress_leave: bool = True,
    mp_context: BaseContext | None = None
) -> list[TaskResult[T, R]]:
    """
    Execute tasks in a process pool with bounded pending task count.
    
    Similar to bounded_thread_pool_map but uses processes instead of threads.
    This is useful for CPU-bound tasks that benefit from true parallelism,
    bypassing Python's GIL.
    
    Special case: If num_workers <= 1, processes tasks sequentially in the
    main thread without spawning any worker processes.
    
    Note:
        - The function `fn` must be picklable (defined at module level).
        - Arguments must also be picklable.
        - Results are returned as they complete, not in order.
    
    Args:
        fn: The function to execute for each item. Must be picklable.
        items: An iterable of (index, args_tuple) pairs. Each args_tuple
            contains the arguments to pass to fn. Args must be picklable.
        num_workers: Number of worker processes. If <= 1, uses sequential processing.
        max_pending: Maximum number of tasks to keep pending at once.
            Controls memory usage. Ignored if num_workers <= 1.
        total: Total number of items for progress bar. If None, tries len(items).
        progress_bar: Whether to show progress bar. Default: True.
        progress_desc: Description for progress bar. Default: None.
        progress_position: Position for nested progress bars. Default: None.
        progress_leave: Whether to leave progress bar after completion. Default: True.
        mp_context: Optional multiprocessing context to use (e.g., mp_get_context('spawn')). Default: None (uses default context).
    
    Returns:
        List of TaskResult objects containing the index, result, and success status.
    
    Example:
        >>> def cpu_intensive(x):
        ...     return sum(i*i for i in range(x))
        >>> items = [(i, (i * 1000,)) for i in range(100)]
        >>> results = bounded_process_pool_map(
        ...     cpu_intensive, items, num_workers=4, max_pending=50
        ... )
    """
    # Determine total for progress bar
    if total is None:
        try:
            total = len(items)  # type: ignore
        except TypeError:
            total = None
    
    # Special case: sequential processing for num_workers <= 1
    if num_workers <= 1:
        results: list[TaskResult[T, R]] = []
        with tqdm(
            total=total,
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            disable=not progress_bar,
        ) as pbar:
            for index, args in items:
                try:
                    result = fn(*args)
                    task_result = TaskResult(index=index, result=result, is_success=True)
                except Exception as e:
                    task_result = TaskResult(index=index, result=e, is_success=False)
                
                results.append(task_result)
                pbar.update(1)
        
        return results
    
    # Parallel processing with process pool
    items_iter = iter(items)
    results: list[TaskResult[T, R]] = []
    
    with tqdm(
        total=total,
        desc=progress_desc,
        position=progress_position,
        leave=progress_leave,
        disable=not progress_bar,
    ) as pbar:
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=mp_context
        ) as executor:
            pending: dict[Future, T] = {}
            
            # Submit initial batch of tasks
            for _ in range(max_pending):
                try:
                    index, args = next(items_iter)
                    future = executor.submit(fn, *args)
                    pending[future] = index
                except StopIteration:
                    break
            
            # Process results and submit new tasks as old ones complete
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                
                for future in done:
                    idx = pending.pop(future)
                    try:
                        result = future.result()
                        task_result = TaskResult(index=idx, result=result, is_success=True)
                    except Exception as e:
                        # raise e
                        task_result = TaskResult(index=idx, result=e, is_success=False)
                    
                    results.append(task_result)
                    pbar.update(1)
                    
                    # Submit a new task if there are more items
                    try:
                        index, args = next(items_iter)
                        new_future = executor.submit(fn, *args)
                        pending[new_future] = index
                    except StopIteration:
                        pass
    
    return results


class BoundedThreadPoolExecutor:
    """
    A context manager that provides bounded task submission for ThreadPoolExecutor.
    
    This class wraps ThreadPoolExecutor and provides a submit method that
    automatically limits the number of pending tasks.
    
    Special case: If num_workers <= 1, processes tasks sequentially in the
    main thread without spawning any worker threads.
    
    Example:
        >>> with BoundedThreadPoolExecutor(num_workers=4, max_pending=50, total=1000) as pool:
        ...     for i, data in enumerate(large_dataset):
        ...         pool.submit(process_func, i, data)
        ...     for result in pool.results():
        ...         print(result.index, result.result)
        
        # With nested progress bars:
        >>> for epoch in tqdm(range(10), desc="Epochs", position=0):
        ...     with BoundedThreadPoolExecutor(
        ...         num_workers=4, max_pending=50, total=1000,
        ...         progress_desc="Processing", progress_position=1, progress_leave=False
        ...     ) as pool:
        ...         for i, data in enumerate(large_dataset):
        ...             pool.submit(process_func, i, data)
    """
    
    def __init__(
        self,
        num_workers: int,
        max_pending: int,
        total: int | None = None,
        progress_bar: bool = True,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        progress_leave: bool = True,
    ):
        """
        Initialize the bounded thread pool executor.
        
        Args:
            num_workers: Number of worker threads. If <= 1, uses sequential processing.
            max_pending: Maximum number of tasks to keep pending at once. Ignored if num_workers <= 1.
            total: Total number of tasks for progress bar. Default: None.
            progress_bar: Whether to show progress bar. Default: True.
            progress_desc: Description for progress bar. Default: None.
            progress_position: Position for nested progress bars. Default: None.
            progress_leave: Whether to leave progress bar after completion. Default: True.
        """
        self.num_workers = num_workers
        self.max_pending = max_pending
        self.total = total
        self.progress_bar = progress_bar
        self.progress_desc = progress_desc
        self.progress_position = progress_position
        self.progress_leave = progress_leave
        self._executor: ThreadPoolExecutor | None = None
        self._pending: dict[Future, Any] = {}
        self._results: list[TaskResult] = []
        self._pbar: tqdm | None = None
        self._sequential_mode = num_workers <= 1
    
    def __enter__(self):
        if not self._sequential_mode:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._pbar = tqdm(
            total=self.total,
            desc=self.progress_desc,
            position=self.progress_position,
            leave=self.progress_leave,
            disable=not self.progress_bar,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Collect remaining results
        if not self._sequential_mode:
            self._drain_pending()
        if self._pbar:
            self._pbar.close()
        if self._executor:
            self._executor.shutdown(wait=True)
        return False
    
    def submit(self, fn: Callable, index: Any, *args, **kwargs) -> None:
        """
        Submit a task, blocking if max_pending limit is reached.
        
        Args:
            fn: The function to execute.
            index: An identifier for this task (returned with results).
            *args, **kwargs: Arguments to pass to fn.
        """
        if self._sequential_mode:
            # Sequential processing: execute immediately
            try:
                result = fn(*args, **kwargs)
                task_result = TaskResult(index=index, result=result, is_success=True)
            except Exception as e:
                task_result = TaskResult(index=index, result=e, is_success=False)
            
            self._results.append(task_result)
            if self._pbar:
                self._pbar.update(1)
        else:
            # Parallel processing with thread pool
            if self._executor is None:
                raise RuntimeError("BoundedThreadPoolExecutor must be used as context manager")
            
            # If at capacity, wait for at least one task to complete
            while len(self._pending) >= self.max_pending:
                self._collect_one()
            
            future = self._executor.submit(fn, *args, **kwargs)
            self._pending[future] = index
    
    def _collect_one(self) -> None:
        """Wait for one task to complete and store its result."""
        if not self._pending:
            return
        
        done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        for future in done:
            idx = self._pending.pop(future)
            try:
                result = future.result()
                task_result = TaskResult(index=idx, result=result, is_success=True)
            except Exception as e:
                task_result = TaskResult(index=idx, result=e, is_success=False)
            
            self._results.append(task_result)
            if self._pbar:
                self._pbar.update(1)
    
    def _drain_pending(self) -> None:
        """Collect all remaining pending results."""
        while self._pending:
            self._collect_one()
    
    def results(self) -> list[TaskResult]:
        """
        Return all collected results.
        
        Note: This should be called after all tasks are submitted and
        the context manager has exited (or after manually calling _drain_pending).
        """
        if not self._sequential_mode:
            self._drain_pending()
        return self._results


class BoundedProcessPoolExecutor:
    """
    A context manager that provides bounded task submission for ProcessPoolExecutor.
    
    Similar to BoundedThreadPoolExecutor but uses processes for CPU-bound tasks.
    
    Special case: If num_workers <= 1, processes tasks sequentially in the
    main thread without spawning any worker processes.
    
    Note:
        - Functions and arguments must be picklable.
    
    Example:
        >>> with BoundedProcessPoolExecutor(num_workers=4, max_pending=50, total=1000) as pool:
        ...     for i, data in enumerate(large_dataset):
        ...         pool.submit(process_func, i, data)
        ...     for result in pool.results():
        ...         print(result.index, result.result)
        
        # With nested progress bars:
        >>> for epoch in tqdm(range(10), desc="Epochs", position=0):
        ...     with BoundedProcessPoolExecutor(
        ...         num_workers=4, max_pending=50, total=1000,
        ...         progress_desc="Processing", progress_position=1, progress_leave=False
        ...     ) as pool:
        ...         for i, data in enumerate(large_dataset):
        ...             pool.submit(process_func, i, data)
    """
    
    def __init__(
        self,
        num_workers: int,
        max_pending: int,
        total: int | None = None,
        progress_bar: bool = True,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        progress_leave: bool = True,
        mp_context: BaseContext | None = None
    ):
        """
        Initialize the bounded process pool executor.
        
        Args:
            num_workers: Number of worker processes. If <= 1, uses sequential processing.
            max_pending: Maximum number of tasks to keep pending at once. Ignored if num_workers <= 1.
            total: Total number of tasks for progress bar. Default: None.
            progress_bar: Whether to show progress bar. Default: True.
            progress_desc: Description for progress bar. Default: None.
            progress_position: Position for nested progress bars. Default: None.
            progress_leave: Whether to leave progress bar after completion. Default: True.
            mp_context: Optional multiprocessing context. Ignored if num_workers <= 1.
        """
        self.num_workers = num_workers
        self.max_pending = max_pending
        self.total = total
        self.progress_bar = progress_bar
        self.progress_desc = progress_desc
        self.progress_position = progress_position
        self.progress_leave = progress_leave
        self.mp_context = mp_context
        self._executor: ProcessPoolExecutor | None = None
        self._pending: dict[Future, Any] = {}
        self._results: list[TaskResult] = []
        self._pbar: tqdm | None = None
        self._sequential_mode = num_workers <= 1
    
    def __enter__(self):
        if not self._sequential_mode:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers, mp_context=self.mp_context
            )
        self._pbar = tqdm(
            total=self.total,
            desc=self.progress_desc,
            position=self.progress_position,
            leave=self.progress_leave,
            disable=not self.progress_bar,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Collect remaining results
        if not self._sequential_mode:
            self._drain_pending()
        if self._pbar:
            self._pbar.close()
        if self._executor:
            self._executor.shutdown(wait=True)
        return False
    
    def submit(self, fn: Callable, index: Any, *args, **kwargs) -> None:
        """
        Submit a task, blocking if max_pending limit is reached.
        
        Args:
            fn: The function to execute (must be picklable if using multiprocessing).
            index: An identifier for this task (returned with results).
            *args, **kwargs: Arguments to pass to fn (must be picklable if using multiprocessing).
        """
        if self._sequential_mode:
            # Sequential processing: execute immediately
            try:
                result = fn(*args, **kwargs)
                task_result = TaskResult(index=index, result=result, is_success=True)
            except Exception as e:
                task_result = TaskResult(index=index, result=e, is_success=False)
            
            self._results.append(task_result)
            if self._pbar:
                self._pbar.update(1)
        else:
            # Parallel processing with process pool
            if self._executor is None:
                raise RuntimeError("BoundedProcessPoolExecutor must be used as context manager")
            
            # If at capacity, wait for at least one task to complete
            while len(self._pending) >= self.max_pending:
                self._collect_one()
            
            future = self._executor.submit(fn, *args, **kwargs)
            self._pending[future] = index
    
    def _collect_one(self) -> None:
        """Wait for one task to complete and store its result."""
        if not self._pending:
            return
        
        done, _ = wait(self._pending, return_when=FIRST_COMPLETED)
        for future in done:
            idx = self._pending.pop(future)
            try:
                result = future.result()
                task_result = TaskResult(index=idx, result=result, is_success=True)
            except Exception as e:
                task_result = TaskResult(index=idx, result=e, is_success=False)
            
            self._results.append(task_result)
            if self._pbar:
                self._pbar.update(1)
    
    def _drain_pending(self) -> None:
        """Collect all remaining pending results."""
        while self._pending:
            self._collect_one()
    
    def results(self) -> list[TaskResult]:
        """
        Return all collected results.
        
        Note: This should be called after all tasks are submitted and
        the context manager has exited (or after manually calling _drain_pending).
        """
        if not self._sequential_mode:
            self._drain_pending()
        return self._results
