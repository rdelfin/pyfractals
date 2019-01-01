import numpy as np
from PIL import Image
from scipy.misc import derivative
from sympy import Symbol, lambdify, Pow
from typing import Callable, Tuple


def main():
    x = Symbol("x")
    f = Pow(x, 4) - 1
    fprime = f.diff(x)
    grid = get_grid(-2 - 1.3333j, 2 + 1.3333j, (800, 600))
    color_map = [
        (1 + 0j, (255, 0, 0)),
        (-1 + 0j, (0, 255, 0)),
        (0 + 1j, (0, 0, 255)),
        (0 - 1j, (255, 255, 0)),
    ]
    eps = 1e-3
    num_iters = 200

    vals, iter_counts, prev_vals = find_zeros(
        lambdify(x, f), lambdify(x, fprime), grid, num_iters, eps
    )

    pixels = np.full(np.shape(vals) + (3,), 0)
    print(f"non-zero iter counts: {len(np.where(iter_counts != None))}")
    for x in range(np.shape(vals)[0]):
        for y in range(np.shape(vals)[1]):
            for zero, color in color_map:
                if abs(vals[x, y] - zero) < eps:
                    factor = (
                        0
                        if iter_counts[x, y] is None
                        else shading_factor(vals[x, y], prev_vals[x, y], zero, eps)
                    )
                    print(f"FACTOR({x}, {y}) = {factor}")
                    pixels[x, y] = tuple(val * factor for val in color)
                    break
                pixels[x, y] = (0, 0, 0)

    image = Image.fromarray(pixels.astype(np.uint8))
    image.save("out.bmp")
    image.show()


def shading_factor(
    curr_step: complex, prev_step: complex, zero: complex, stop_threshold: float
) -> float:
    d0 = abs(prev_step - zero)
    d1 = abs(curr_step - zero)
    val = (np.log(stop_threshold) - np.log(d0)) / (np.log(d1) - np.log(d0))
    return val


def get_grid(min: complex, max: complex, dims: Tuple[int, int]) -> np.ndarray:
    separated_mesh = np.meshgrid(
        np.linspace(min.real, max.real, num=dims[0]),
        np.linspace(min.imag, max.imag, num=dims[1]),
    )
    return separated_mesh[0] + separated_mesh[1] * 1j


def find_zeros(
    f: Callable,
    fprime: Callable,
    initial_vals: np.ndarray,
    iters: int = 200,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = initial_vals.copy()
    iter_counts = np.full(np.shape(vals), None)
    prev_vals = np.full(np.shape(vals), None)

    for i in range(iters):
        print(f"ITERATION {i+1}/{iters}")
        new_vals = newton_iter(f, fprime, vals)
        stationary_points = set(zip(*np.where(np.absolute(new_vals - vals) < eps)))
        not_finished_points = set(zip(*np.where(iter_counts == None)))
        # Grab all points that haven't been marked with an iteration count but
        # already stopped
        update_points = stationary_points & not_finished_points

        for point in update_points:
            iter_counts[point] = i
            prev_vals[point] = vals[point]

        vals = new_vals.copy()

    print("DONE ITERS")
    return (vals, iter_counts, prev_vals)


def newton_iter(f: Callable, fprime: Callable, vals: np.ndarray) -> np.ndarray:
    return vals - f(vals) / fprime(vals)


if __name__ == "__main__":
    main()
