import numpy as np
from PIL import Image
from scipy.misc import derivative
from sympy import Symbol, lambdify, Pow
from typing import Callable, Tuple


def main():
    x = Symbol("x")
    f = Pow(x, 4) - 1
    fprime = f.diff(x)
    grid = get_grid(-2 - 2j, 2 + 2j, (2000, 1400))
    color_map = [
        (1 + 0j, (255, 0, 0)),
        (-1 + 0j, (0, 255, 0)),
        (0 + 1j, (0, 0, 255)),
        (0 - 1j, (255, 255, 0)),
    ]

    vals, iter_counts = find_zeros(lambdify(x, f), lambdify(x, fprime), grid)

    pixels = np.full(np.shape(vals) + (3,), 0)
    for x in range(np.shape(vals)[0]):
        for y in range(np.shape(vals)[1]):
            for zero, color in color_map:
                if abs(vals[x, y] - zero) < 1e-6:
                    factor = shading_factor(iter_counts[x, y])
                    pixels[x, y] = tuple(val*factor for val in color)
                    break
                pixels[x, y] = (0, 0, 0)

    image = Image.fromarray(pixels.astype(np.uint8))
    image.save("out.bmp")
    image.show()


def shading_factor(iterations: int) -> float:
    return -2.0/(1.0+np.exp(-0.1*(iterations - 1))) + 2


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
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = initial_vals.copy()
    iter_counts = np.full(np.shape(vals), -1)
    for i in range(iters):
        print(f"ITERATION {i+1}/{iters}")
        new_vals = newton_iter(f, fprime, vals)
        stationary_points = set(zip(*np.where(np.absolute(new_vals - vals) < eps)))
        not_finished_points = set(zip(*np.where(iter_counts < 0)))
        # Grab all points that haven't been marked with an iteration count but
        # already stopped
        update_points = stationary_points & not_finished_points

        for point in update_points:
            iter_counts[point] = i
        vals = new_vals.copy()

    print("DONE ITERS")
    return (vals, iter_counts)


def newton_iter(f: Callable, fprime: Callable, vals: np.ndarray) -> np.ndarray:
    return vals - f(vals) / fprime(vals)


if __name__ == "__main__":
    main()
