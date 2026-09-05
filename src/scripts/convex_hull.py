import time
from typing import List, Tuple, Dict, Optional
import statistics
from dataclasses import dataclass
import random
import numpy as np

@dataclass
class Function:
    x1: float    # start point (function is +inf for x < x1)
    a: float     # slope
    b: float     # y-intercept
    id: str      # unique identifier
    def evaluate(self, x: float) -> float:
        """Evaluate function at point x"""
        if x < self.x1:
            return float('inf')
        return self.a * x + self.b

def generate_random_functions(n: int, 
                            x1_range: Tuple[float, float] = (0, 100),
                            a_range: Tuple[float, float] = (0.1, 10),
                            b_range: Tuple[float, float] = (0, 100)) -> List[Function]:
    """Generate n random stepwise affine functions"""
    functions = []
    for _ in range(n):
        x1 = random.uniform(*x1_range)
        a = random.uniform(*a_range)
        b = random.uniform(*b_range)
        id = f'{x1}_{a}_{b}'
        functions.append(Function(x1, a, b, id))
    return functions

def generate_query_points(n: int, 
                         x_range: Tuple[float, float] = (0, 200)) -> List[float]:
    """Generate n sorted query points"""
    return sorted([random.uniform(*x_range) for _ in range(n)])

def naive_solve(functions: List[Function], query_points: List[float]) -> List[float]:
    """Naive solution - evaluate all functions at each query point"""
    results = []
    for x in query_points:
        min_y = float('inf')
        for f in functions:
            min_y = min(min_y, f.evaluate(x))
        results.append(min_y)
    return results

def intersection_x(f1: Function, f2: Function) -> float:
    """Find x-coordinate of intersection of two lines"""
    # y = a1x + b1 = a2x + b2
    # x = (b2 - b1)/(a1 - a2)
    if abs(f1.a - f2.a) < 1e-10:  # parallel lines
        return float('inf') if f1.b <= f2.b else float('-inf')
        
    return (f2.b - f1.b)/(f1.a - f2.a)

def is_redundant(f1: Function, f2: Function, f3: Function) -> bool:
    """Check if f2 is redundant given f1 and f3"""
    # If intersection of f1,f2 occurs after intersection of f2,f3
    # then f2 is redundant
    x12 = intersection_x(f1, f2)
    x23 = intersection_x(f2, f3)
    return x12 > x23

def binary_search_by_slope(hull: List[Function], new_func: Function) -> int:
    """Find position to insert new function to maintain descending slope order"""
    left, right = 0, len(hull)
    while left < right:
        mid = (left + right) // 2
        if hull[mid].a >= new_func.a:
            left = mid + 1
        else:
            right = mid
    return left

def _precompute_intersections(hull: List[Function]) -> np.ndarray:
    """Precompute intersection x-coords between adjacent hull segments."""
    n = len(hull)
    if n <= 1:
        return np.empty(0, dtype=np.float64)
    pts = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        pts[i] = intersection_x(hull[i], hull[i + 1])
    return pts


def convex_hull_preprocess(functions: List[Function]) -> Tuple[List[List[Function]], List[np.ndarray], List[float]]:
    """Preprocess functions using convex hull trick.

    Returns:
        hulls:       list of hull snapshots (one per function activation)
        hull_isects: precomputed intersection arrays for each hull
        x1_points:   sorted activation thresholds
    """
    events = sorted(functions, key=lambda f: f.x1)
    x1_points = [f.x1 for f in events]

    hulls = []
    hull_isects = []
    current_hull: List[Function] = []

    for new_func in events:
        pos = binary_search_by_slope(current_hull, new_func)
        current_hull.insert(pos, new_func)

        # Check if newly inserted function is itself redundant
        if pos > 0 and pos + 1 < len(current_hull):
            if is_redundant(current_hull[pos-1], current_hull[pos], current_hull[pos+1]):
                current_hull.pop(pos)
                pos -= 1

        # Remove redundant functions to the left
        while pos >= 2:
            if is_redundant(current_hull[pos-2], current_hull[pos-1], current_hull[pos]):
                current_hull.pop(pos-1)
                pos -= 1
            else:
                break

        # Remove redundant functions to the right
        while pos + 2 < len(current_hull):
            if is_redundant(current_hull[pos], current_hull[pos+1], current_hull[pos+2]):
                current_hull.pop(pos+1)
            else:
                break

        # Shallow copy: Function objects are never mutated
        snapshot = current_hull[:]
        hulls.append(snapshot)
        hull_isects.append(_precompute_intersections(snapshot))

    return hulls, hull_isects, x1_points

def binary_search_segment(x1_points: List[float], x: float) -> int:
    """Find rightmost point <= x"""
    left, right = 0, len(x1_points)
    while left < right:
        mid = (left + right) // 2
        if x1_points[mid] <= x:
            left = mid + 1
        else:
            right = mid
    return left - 1

def query_hull_with_id(hull: List[Function], x: float) -> Tuple[float, Function]:
    """Find minimum value and corresponding function at x in convex hull"""
    if not hull:
        return float('inf'), None
    
    # If only one function
    if len(hull) == 1:
        return hull[0].evaluate(x), hull[0]

    # Binary search to find the correct segment
    left, right = 0, len(hull) - 1
    while left < right:
        mid = (left + right) // 2
        
        # Get intersection points around mid
        t1 = float('-inf') if mid == 0 else intersection_x(hull[mid-1], hull[mid])
        t2 = float('inf') if mid == len(hull)-1 else intersection_x(hull[mid], hull[mid+1])
        
        if t1 <= x <= t2:
            # x is in this segment, evaluate mid function
            return hull[mid].evaluate(x), hull[mid]
        elif x < t1:
            # x is in left half
            right = mid
        else:
            # x is in right half
            left = mid + 1
            
    # If we get here, evaluate the last function
    return hull[left].evaluate(x), hull[left]

def query_hull(hull: List[Function], x: float) -> float:

    return query_hull_with_id(hull,x)[0]


def query_hull_vectorized(
    hull: List[Function],
    isects: np.ndarray,
    query_xs: np.ndarray,
) -> Tuple[np.ndarray, List[Optional[Function]]]:
    """Evaluate minimum over hull for all query points at once.

    Uses precomputed intersections and np.searchsorted for O(Q log H)
    vectorized evaluation instead of Q individual Python binary searches.
    """
    Q = len(query_xs)
    if not hull:
        return np.full(Q, np.inf), [None] * Q

    H = len(hull)
    if H == 1:
        f = hull[0]
        vals = np.where(query_xs >= f.x1, f.a * query_xs + f.b, np.inf)
        return vals, [f if query_xs[i] >= f.x1 else None for i in range(Q)]

    # np.searchsorted: find which hull segment each query falls into
    seg_indices = np.searchsorted(isects, query_xs, side='right')
    seg_indices = np.clip(seg_indices, 0, H - 1)

    # Vectorized evaluation: y = a * x + b
    a_arr = np.array([hull[j].a for j in seg_indices])
    b_arr = np.array([hull[j].b for j in seg_indices])
    values = a_arr * query_xs + b_arr

    funcs = [hull[j] for j in seg_indices]
    return values, funcs


def optimized_solve(functions: List[Function], query_points: List[float]) -> List[float]:
    """Solve using convex hull trick"""
    # Preprocess
    hulls, hull_isects, x1_points = convex_hull_preprocess(functions)

    # Process queries
    results = []
    for x in query_points:
        segment = binary_search_segment(x1_points, x)
        if segment < 0:
            results.append(float('inf'))
        else:
            results.append(query_hull(hulls[segment], x))
    return results

def run_test(n_functions: int = 100, n_queries: int = 50) -> bool:
    """Run test comparing naive and optimized solutions"""
    print(f"\nTesting with {n_functions} functions and {n_queries} queries...")
    
    # Generate test data
    functions = generate_random_functions(n_functions)
    query_points = generate_query_points(n_queries)
    
    print("\nGenerated Functions:")
    for i, f in enumerate(functions):  
        print(f"Function {i}: y = {f.a:.2f}x + {f.b:.2f} for x >= {f.x1:.2f}")
        
    print("\nQuery Points:", query_points)
    
    # Get results from both methods
    print("\nComputing results...")
    naive_results = naive_solve(functions, query_points)
    optimized_results = optimized_solve(functions, query_points)
    
    print("\nResults comparison:")
    for i in range(len(query_points)):
        print(f"x={query_points[i]:.2f}: naive={naive_results[i]:.2f}, optimized={optimized_results[i]:.2f}")
    
    # Compare results
    max_diff = max(abs(n - o) for n, o in zip(naive_results, optimized_results) if n != float('inf') and o != float('inf'))
    if max_diff < 1e-10:
        print("Test passed! Maximum difference:", max_diff)
        return True
    else:
        print("Test failed! Maximum difference:", max_diff)
        return False

def time_solution(functions: List[Function], query_points: List[float], method: str) -> Tuple[List[float], float]:
    """Time either naive or optimized solution"""
    start_time = time.time()
    if method == 'naive':
        results = naive_solve(functions, query_points)
    else:
        results = optimized_solve(functions, query_points)
    elapsed = time.time() - start_time
    return results, elapsed

def run_timing_test(n_functions: int, n_queries: int, n_trials: int = 3) -> Dict:
    """Run timing test with given configuration multiple times"""
    naive_times = []
    optimized_times = []
    max_diffs = []
    
    for trial in range(n_trials):
        # Generate test data
        functions = generate_random_functions(n_functions)
        query_points = generate_query_points(n_queries)
        
        # Time both methods
        naive_results, naive_time = time_solution(functions, query_points, 'naive')
        optimized_results, optimized_time = time_solution(functions, query_points, 'optimized')
        
        # Record times
        naive_times.append(naive_time)
        optimized_times.append(optimized_time)
        
        # Check accuracy
        max_diff = max(abs(n - o) for n, o in zip(naive_results, optimized_results) 
                      if n != float('inf') and o != float('inf'))
        max_diffs.append(max_diff)
    
    return {
        'n_functions': n_functions,
        'n_queries': n_queries,
        'naive_time_avg': statistics.mean(naive_times),
        'naive_time_std': statistics.stdev(naive_times) if len(naive_times) > 1 else 0,
        'optimized_time_avg': statistics.mean(optimized_times),
        'optimized_time_std': statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0,
        'max_error': max(max_diffs),
        'speedup': statistics.mean(naive_times) / statistics.mean(optimized_times)
    }

def run_comprehensive_comparison():
    """Run comprehensive timing comparison with different configurations"""
    configs = [
        (100, 100),     # small case
        (1000, 100),    # more functions
        (100, 1000),    # more queries
        (1000, 1000),   # medium case
        (10000, 100),   # many functions
        (1000, 10000),  # many queries
        (10000, 10000),  # equal
    ]
    
    results = []
    for n_func, n_query in configs:
        print(f"\nTesting with {n_func} functions and {n_query} queries...")
        result = run_timing_test(n_func, n_query)
        results.append(result)
        
        print(f"Configuration: {n_func} functions, {n_query} queries")
        print(f"Naive time: {result['naive_time_avg']:.4f} ± {result['naive_time_std']:.4f} seconds")
        print(f"Optimized time: {result['optimized_time_avg']:.4f} ± {result['optimized_time_std']:.4f} seconds")
        print(f"Speedup factor: {result['speedup']:.2f}x")
        print(f"Max error: {result['max_error']:.2e}")
    
    return results

def optimized_solve_with_id(functions: List[Function], query_points: List[float]) -> List[Tuple[float, Optional[Function]]]:
    """Solve using convex hull trick with vectorized queries.

    Returns List[(min_energy, Function)] for each query point.
    Uses np.searchsorted for batch evaluation instead of per-query binary search.
    """
    if not functions:
        return [(float('inf'), None)] * len(query_points)

    hulls, hull_isects, x1_points = convex_hull_preprocess(functions)
    x1_arr = np.array(x1_points)
    query_arr = np.array(query_points, dtype=np.float64)
    Q = len(query_arr)

    # For each query, find which hull snapshot to use
    snapshot_indices = np.searchsorted(x1_arr, query_arr, side='right') - 1

    result_values = np.full(Q, np.inf)
    result_funcs: List[Optional[Function]] = [None] * Q

    # Group queries by snapshot and batch-evaluate
    unique_snaps = np.unique(snapshot_indices)
    for snap_idx in unique_snaps:
        if snap_idx < 0:
            continue
        mask = snapshot_indices == snap_idx
        query_subset = query_arr[mask]

        vals, funcs = query_hull_vectorized(hulls[snap_idx], hull_isects[snap_idx], query_subset)
        positions = np.where(mask)[0]
        result_values[positions] = vals
        for i, pos in enumerate(positions):
            result_funcs[pos] = funcs[i]

    return list(zip(result_values.tolist(), result_funcs))

def naive_solve_with_id(functions: List[Function], query_points: List[float]) -> List[Tuple[float, Function]]:
    """Naive solution that returns both minimum value and corresponding function"""
    results = []
    for x in query_points:
        min_y = float('inf')
        min_func = None
        for f in functions:
            y = f.evaluate(x)
            if y < min_y:
                min_y = y
                min_func = f
        results.append((min_y, min_func.id))
    return results

def run_test_with_id(n_functions: int = 100, n_queries: int = 50) -> bool:
    """Run test comparing naive and optimized solutions with function identification"""
    print(f"\nTesting with {n_functions} functions and {n_queries} queries...")
    
    # Generate test data
    functions = generate_random_functions(n_functions)
    query_points = generate_query_points(n_queries)
    
    print("\nGenerated Functions:")
    for i, f in enumerate(functions):  
        print(f"Function {i}: y = {f.a:.2f}x + {f.b:.2f} for x >= {f.x1:.2f}")
        
    print("\nQuery Points:", query_points)
    
    # Get results from both methods
    print("\nComputing results...")
    naive_results = naive_solve_with_id(functions, query_points)
    optimized_results = optimized_solve_with_id(functions, query_points)
    
    print("\nResults comparison:")
    all_correct = True
    for i, (x, (naive_val, naive_func), (opt_val, opt_func)) in enumerate(zip(query_points, naive_results, optimized_results)):
        print(f"\nQuery point x={x:.2f}:")
        
        if naive_func is None and opt_func is None:
            print("  Both methods: No active function (inf)")
            continue
            
        print(f"  Naive:     value={naive_val:.2f} using {naive_func}")
        print(f"  Optimized: value={opt_val:.2f} using {opt_func}")
        
        # Check if results match
        if abs(naive_val - opt_val) > 1e-10:
            print(f"  WARNING: Values differ by {abs(naive_val - opt_val):.2e}")
            all_correct = False
        
        # Check if same function or different function giving same value
        if naive_func != opt_func:
            if abs(naive_val - opt_val) <= 1e-10:
                print("  Note: Different functions giving same minimum value")
            else:
                print("  WARNING: Different functions selected")
                all_correct = False
    
    if all_correct:
        print("\nTest passed! All results match.")
    else:
        print("\nTest failed! Some discrepancies found.")
    
    return naive_results, optimized_results


def get_envelope_breakpoints(hulls, hull_isects, x1_points):
    """Extract all breakpoints of the lower envelope.

    Breakpoints are x-values where the piecewise-linear lower envelope
    changes slope: activation points (x1) and hull segment intersections.

    Returns:
        Sorted list of unique breakpoints.
    """
    breakpoints = set()
    n = len(hulls)
    for i in range(n):
        x_lo = x1_points[i]
        x_hi = x1_points[i + 1] if i + 1 < n else float('inf')
        breakpoints.add(x_lo)
        for ix in hull_isects[i]:
            if np.isfinite(ix) and ix >= x_lo and (x_hi == float('inf') or ix < x_hi):
                breakpoints.add(ix)
    return sorted(breakpoints)


def convex_hull_min_e_multi_group_adaptive(group_parsed_results: Dict[int, List[Tuple[float, float, float, str]]]):
    """Adaptive convex hull solver across multiple fusion groups.

    Key insight: since all slopes (static_power) are positive, the summed
    envelope is monotonically increasing between breakpoints. Drops only
    occur at x1 activation points (where a new cheaper configuration becomes
    available). Therefore only x1 points need to be evaluated for energy.

    For EDP = energy * latency (piecewise-quadratic), the true minimum could
    also be at the interior of a segment. We compute the analytical quadratic
    minimum for each segment and add those as extra query points.

    Args:
        group_parsed_results: group_idx -> List[(latency, static_power, dynamic_energy, id)]

    Returns:
        (query_points, group_results_dict)
        query_points: sorted list of critical x-values
        group_results_dict: group_idx -> [(min_e, func), ...] at each query point
    """
    all_x1 = set()
    group_preprocessed = {}

    for group_idx, parsed_results in group_parsed_results.items():
        if not parsed_results:
            continue
        functions = [Function(x, y, z, fid) for x, y, z, fid in parsed_results]
        preprocessed = convex_hull_preprocess(functions)
        group_preprocessed[group_idx] = preprocessed
        _, _, x1_points = preprocessed
        all_x1.update(x1_points)

    if not all_x1:
        return [], {}

    query_points = sorted(all_x1)
    query_arr = np.array(query_points, dtype=np.float64)
    Q = len(query_arr)

    group_results = {}
    for group_idx, preprocessed in group_preprocessed.items():
        hulls, hull_isects, x1_points = preprocessed
        x1_arr = np.array(x1_points)

        snapshot_indices = np.searchsorted(x1_arr, query_arr, side='right') - 1
        result_values = np.full(Q, np.inf)
        result_funcs: List[Optional[Function]] = [None] * Q

        unique_snaps = np.unique(snapshot_indices)
        for snap_idx in unique_snaps:
            if snap_idx < 0:
                continue
            mask = snapshot_indices == snap_idx
            query_subset = query_arr[mask]
            vals, funcs = query_hull_vectorized(hulls[snap_idx], hull_isects[snap_idx], query_subset)
            positions = np.where(mask)[0]
            result_values[positions] = vals
            for i, pos in enumerate(positions):
                result_funcs[pos] = funcs[i]

        group_results[group_idx] = list(zip(result_values.tolist(), result_funcs))

    return query_points, group_results


def convex_hull_min_e(parsed_result:List[Tuple[float, float, float, str]], query_points: List[float]) -> bool:
    """"For a single layer in a network"""
    """Input: parsed_result:List[Tuple[float, float, float, str]]: List[(latency,static power, dynamic energy, id (arch+mapping_id))]"""
    """Input: query_points: List[float] latency of interested"""
    """                                         """
    """Return List[ (min_e,func) for each freq] """
    # Generate test data
    functions = []
    for index, (x, y, z, id) in enumerate(parsed_result):

        functions.append(Function(x, y, z, id))

    optimized_results = optimized_solve_with_id(functions, query_points)

    return optimized_results


def prepare_convex_hull_input(results_manager, 
                            net: str, 
                            layer: str,
                            query_points: List[float],
                            cycle_time: float) -> Tuple[List[Tuple[float, float, float, str]], List[float]]:
    """Prepare input for convex_hull_min_e function."""
    parsed_results = []
    
    for chiplet_id in results_manager.results[net][layer]:
        mappings = results_manager.results[net][layer][chiplet_id]
        for mapping_idx, result in mappings.items():
            if result:
                latency, static_power, dynamic_energy = result.get_metrics(cycle_time)
                # Create identifier for this configuration including mapping
                config_id = f"{chiplet_id}_mapping{mapping_idx}"
                parsed_results.append((latency, static_power, dynamic_energy, config_id))
    
    return parsed_results, query_points

def process_layer_convex_hull(results_manager,
                            net: str,
                            layer: str,
                            query_points: List[float],
                            cycle_time: float) -> Dict:
    """
    Process a single layer using convex hull optimization.
    
    Args:
        results_manager: ResultsManager containing all mapping results
        net: Network name
        layer: Layer name
        cycle_time: Current cycle time in seconds

    Returns:
    List[ (min_e,func) for each freq] 
    """
    # Prepare input
    parsed_results, query_points = prepare_convex_hull_input(
        results_manager, net, layer, cycle_time, query_points
    )
    
    # Skip if no valid results
    if not parsed_results:
        print(f"No valid results for {net} layer {layer}")
        return None
    
    # Call convex hull optimization
    return convex_hull_min_e(parsed_results, query_points)


if __name__ == "__main__":
    # Run a test
    naive_results, optimized_results = run_test_with_id(5, 3)


