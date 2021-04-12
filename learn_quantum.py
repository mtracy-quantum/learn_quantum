from math import pi, sqrt, log, log2, floor
import fractions as frac
from cmath import phase

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

from qiskit import Aer, IBMQ, execute

from qiskit.tools.monitor import job_monitor
from qiskit.quantum_info.operators.pauli import Pauli
from itertools import groupby
from itertools import product as iter_product
import qiskit.quantum_info.synthesis.two_qubit_decompose as twoq
import matplotlib.pyplot as plt
from sympy import log as sympy_log, re, im, acos, atan, sin, cos, factorint, \
    primefactors, gcd, mod_inverse, prime

from IPython.display import Latex, display

imag = complex(0, 1)

ket_0 = np.array([[1], [0]])
ket_1 = np.array([[0], [1]])

ket_plus = np.array([[1 / sqrt(2)], [1 / sqrt(2)]])
ket_minus = np.array([[1 / sqrt(2)], [- 1 / sqrt(2)]])

ket_l = np.array([[1 / sqrt(2)], [imag / sqrt(2)]])
ket_r = np.array([[1 / sqrt(2)], [-imag / sqrt(2)]])

pauliX = np.array([[0, 1], [1, 0]], dtype=complex)
pauliY = np.array([[0, 0. - 1.j], [0. + 1.j, 0]], dtype=complex)
pauliZ = np.array([[1, 0], [0, -1]], dtype=complex)

hadamard = (1 / sqrt(2)) * np.array([[1 + 0.j, 1 + 0.j],
                                     [1 + 0.j, -1 + 0.j]], dtype=complex)

cnot01 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)

cnot10 = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]], dtype=complex)

swap = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)

# line feed in latex
llf = r'\begin{equation} \\ \end{equation}'

default_bracket_type = 'p'


def ket(str_bits):
    ret = get_basis(str_bits[0])
    for n in range(1, len(str_bits)):
        ret = np.kron(ret, get_basis(str_bits[n]))
    return ret


def bra(str_bits):
    return ket(str_bits).T.conj()


def get_basis(char_bit):
    if char_bit == '0' or char_bit == 'H':
        return ket_0
    if char_bit == '1' or char_bit == 'V':
        return ket_1
    if char_bit == '+':
        return ket_plus
    if char_bit == '-':
        return ket_minus
    if char_bit == 'L':
        return ket_l
    if char_bit == 'R':
        return ket_r

    raise ValueError('Invalid character passed to get_basis')


def initialize_register(circuit, register, value, reverse=True):
    """
    Classical binary initialization
    e.g. value=3 sets '00011' depending on size of register
    :param circuit: the circuit you wish to initialize
    :param register: register in the circuit
    :param value: integer value to set the binary values
    :param reverse: reverse the qiskit format
    :return: none
    """
    for k in range(len(register)):
        if 2 ** k & value:
            if reverse:
                circuit.x(register[len(register) - k - 1])
            else:
                circuit.x(register[k])


def results_by_qubit(answers, reverse=True):
    """
    Breaks the result set down by qubit and returns how many times each qubit
    had a result of 0 and 1.
    :param answers: The result set to break down
    :param reverse: Whether to reverse Qiskit format
    :return:  lists of qubit values of zeros and ones
    """
    bit_size = len(next(iter(answers)))
    zeros = [0] * bit_size
    ones = [0] * bit_size

    for val in answers:
        n = 0
        if reverse:
            loop_string = reverse_string(val)
        else:
            loop_string = val
        for bit in loop_string:
            if bit == '0':
                zeros[n] += answers[val]
            elif bit == '1':
                ones[n] += answers[val]
            ## it could be a space for split registers, leave as 0
            n += 1
    return zeros, ones


def combine_results(a_base, a_new):
    """
    Takes two results and adds the values for like keys together

    :param a_base: A results set ( the one you are adding to)
    :param a_new: A result set (typically the one you just ran and are adding to the total)
    :return:  the a_base value with the a_new values added to it
    """
    for key, value in a_new.items():
        if key in a_base:
            a_base[key] += value
        else:
            a_base[key] = value
    return a_base


def format_results(results, integer=False, threshold=0, reverse=True, split=None):
    """
    Formats results by register, converts to integers, and/or reverses bits
    So 000010 01: 200  can be formated as 2 1: 200.

    :param results: The results to split
    :param integer: Format results as integer
    :param threshold: Drop results with fewer than theshold values (not percentage).
    :param reverse: Reverse from Qiskit standard
    :param split:
    :return:
    """

    new_results = {}
    for k, v in results.items():
        if reverse:
            k = reverse_string(k)
        if integer:
            new = ''
            for val in k.split(' '):
                new += str(int(val, 2)) + '  '
            k = new
        if v > threshold:
            new_results[k] = v
    return new_results.items()


def get_axis(axis, qr, cr):
    """ Returns a QuantumCircuit given a string such as X, XX, XYZ"""
    axis = axis.upper()
    size = len(axis)

    a = QuantumCircuit(qr, cr)
    for n in range(size):
        if axis[n] == 'X':
            a.h(qr[n])
        if axis[n] == 'Y':
            a.sdg(qr[n])
            a.h(qr[n])
    return a


def get_measure(axis, qr, cr):
    """
    For X, Y, Z axes, construct a QuantumCircuit that measures a single QuantumRegister
    :param axis: Axes to be measured, can be multiple.  e.g. 'X' , 'XYZ' -- one for each qubit
    :param qr: QuantumRegister to measure
    :param cr: ClassicalRegister to measure into
    :return: QuantumCircuit consisting of measurement
    """
    m = get_axis(axis, qr, cr)
    m.measure(qr, cr)
    return m


def print_measurements(axes, circuit, qr, cr, shots=1000, seed_simulator=None):
    size = qr.size
    array_axes = generate_axes(axes.upper(), size)

    for axis in array_axes:
        answers = execute_simulated(
            circuit + get_measure(axis, qr, cr),
            shots, seed_simulator=seed_simulator)
        print_reverse_results(answers, label=axis)


def generate_axes(axes, count):
    """ Returns an array of strings ['XX', 'XZ',...]
    It splits up the individual characters of axes and permutates count times
    So ('XY', 2) returns XX, XY, YX, YY.

    Keyword arguments:
    axis -- a string of any combination of X, Y, Z -- 'XY', 'XYZ'
    count -- the number to permutate over.

    Returns a string count characters long.
    """
    array_axes = []
    all_axes = iter_product(axes, repeat=count)
    for b in all_axes:
        array_axes.append(''.join(str(i) for i in b))
    return array_axes


def print_reverse_results(results, label=None, integer=False, threshold=0):
    lbl = 'Reversed:'
    if label is not None:
        lbl = lbl + label + ':'
    print(lbl, sorted(format_results(results, integer=integer, threshold=threshold, reverse=True)))


def print_results(results, label=None, integer=False, reverse=False, threshold=0):
    lbl = ''
    if label is not None:
        lbl = lbl + label + ':'
    print(lbl, sorted(format_results(results, integer=integer, threshold=threshold, reverse=reverse)))


def swap_rows(arr, row_1, row_2):
    arr[[row_1, row_2], :] = arr[[row_2, row_1], :]


def swap_columns(arr, col_1, col_2):
    arr[:, [col_1, col_2]] = arr[:, [col_2, col_1]]


def double_swap_map(old):
    # First map each entire onto 2x its previous mapping
    old[::, 1:2] = old[::, 1:2] * 2
    # Then add a mapping for the new entries
    # copy it to the end and add 1 to the mapping
    rows = old.shape[0]
    for k in range(rows):
        old = np.append(old, [[old[k, 0] + rows, old[k, 1] + 1]], axis=0)
    return old


def swap_entries(qiskit_array):
    size = qiskit_array.shape[0]
    bit_size = int(log2(size))

    swap_array = np.array([[0, 0], [1, 1]])

    for k in range(bit_size - 1):
        swap_array = double_swap_map(swap_array)

    for map_vals in swap_array:
        if map_vals[1] > map_vals[0]:
            swap_columns(qiskit_array, map_vals[0], map_vals[1])
            swap_rows(qiskit_array, map_vals[0], map_vals[1])

    return qiskit_array


def print_matrix(qc):
    print('Adjusted Matrix:')
    with np.printoptions(linewidth=1024):
        print(what_is_the_matrix(qc))


def print_eigen_periods(qc):
    if isinstance(qc, QuantumCircuit):
        unitary = what_is_the_matrix(qc)
    else:
        unitary = qc
    w, v = np.linalg.eig(unitary)

    periods = []

    for val in w:
        rounded = complex(round(val.real, 8), round(val.imag, 8))
        exponent = sympy_log(rounded)
        fraction = get_rotation_fraction(float(im(exponent)), positive_only=True)

        period = 2*fraction.denominator

        if not round(float(re(exponent)), 4) == 0:
            print(val)
        else:
            if period not in periods:
                periods.append(period)
    print(','.join([str(p) for p in sorted(periods)]))
    return periods


def show_eigen_values(qc, display_distinct=False, display_exp=False, display_omega=False, omega_size=0):
    if isinstance(qc, QuantumCircuit):
        unitary = what_is_the_matrix(qc)
    else:
        unitary = qc
    w, v = np.linalg.eig(unitary)

    if omega_size == 0:
        omega_size = v[0].shape[0]

    display_vals = w

    if display_distinct:
        distinct_vals = []
        for val in w:
            rounded = complex(round(val.real, 8), round(val.imag, 8))
            if not rounded in distinct_vals:
                distinct_vals.append(rounded)
        display_vals = distinct_vals

    output = r'\begin{equation*}'
    for n in range(len(display_vals)):
        if display_distinct:
            index_vals = []
            for i, val in enumerate(w):
                rounded = complex(round(val.real, 8), round(val.imag, 8))
                if rounded == display_vals[n]:
                    index_vals.append(i)
            index_string = ','.join([str(y) for y in index_vals])
        else:
            index_string = str(n)

        if display_exp:
            output += r'\lambda_{' + index_string + r'}=' + format_complex_as_exponent(display_vals[n])
        else:
            output += r'\lambda_{' + index_string + r'}=' + format_complex_as_latex(display_vals[n])
        if display_omega:
            output += r'=' + format_complex_as_omega(w[n], omega_size=omega_size)

        output += r',\; '

        output += r'\quad'
    output += r'\end{equation*}'

    display(Latex(output))


def show_eigens(qc, bracket_type=None, display_exp=False, display_omega=False, omega_size=0):
    if isinstance(qc, QuantumCircuit):
        unitary = what_is_the_matrix(qc)
    else:
        unitary = qc
    w, v = np.linalg.eig(unitary)

    if omega_size == 0:
        omega_size = v[0].shape[0]

    bracket_type = get_bracket_type(bracket_type)
    output = r'\begin{equation*}'
    for n in range(w.shape[0]):
        if display_exp:
            output += r'\lambda_{' + str(n) + r'}=' + format_complex_as_exponent(w[n])
        else:
            output += r'\lambda_{' + str(n) + r'}=' + format_complex_as_latex(w[n])
        if display_omega:
            output += r'=' + format_complex_as_omega(w[n], omega_size=omega_size)
        output += r',\; '
        output += np_array_to_latex(v[:, n].reshape(v[:, n].shape[0], 1),
                                    bracket_type=bracket_type, factor_out=True, begin_equation=False,
                                    display_exp=display_exp, display_omega=display_omega, omega_size=omega_size,
                                    label='v_{' + str(n) +'}')
        output += r'\quad'
    output += r'\end{equation*}'

    display(Latex(output))


def what_is_the_matrix(qc):
    qiskit_array = execute_unitary(qc)
    return swap_entries(qiskit_array)


def show_me_the_matrix(qc, bracket_type=None, factor_out=True, max_display_size=16,
                       normalize=False, label=None, display_exp=False, display_omega=False, omega_size=0):
    if isinstance(qc, QuantumCircuit):
        unitary = what_is_the_matrix(qc)
    else:
        unitary = qc

    # limit the size
    truncated_str = ''
    if omega_size == 0:
        omega_size = unitary.shape[0]
    if unitary.shape[0] > max_display_size:
        unitary = unitary[0:max_display_size-1, 0:max_display_size-1]
        truncated_str = r'Max Display Size Exceeded'
    display(Latex(np_array_to_latex(unitary,
                                    bracket_type=get_bracket_type(bracket_type),
                                    factor_out=factor_out,
                                    normalize=normalize,
                                    label=label,
                                    display_exp=display_exp, display_omega=display_omega, omega_size=omega_size) + truncated_str))


def what_is_the_state_vector(qc):
    state_vector = execute_state_vector(qc)
    return state_vector.reshape(state_vector.shape[0], 1)


def what_is_the_density_matrix(qc):
    state_vector = what_is_the_state_vector(qc)
    return state_vector @ state_vector.T.conj()


def show_density_matrix(qc, bracket_type=None, factor_out=False, label=None):
    state_vector = execute_state_vector(qc)
    sv = state_vector.reshape(1, state_vector.shape[0])
    density_matrix = sv.T.conj() @ sv
    if not np.isclose(np.trace(density_matrix @ density_matrix), 1):
        return 'Not a pure state -- not implemented for mixed'

    display(Latex(np_array_to_latex(density_matrix,
                                    bracket_type=get_bracket_type(bracket_type),
                                    factor_out=factor_out,
                                    label=label)))


def permutation_integers(mat):
    """
    Return list of integers representing one-hot rows from a permutation matrix
    :param mat: Permutation matrix (no checking)
    :return: list of integers
    """
    ret = []
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            # When multipled by a one-hot ket,
            # the column is returned corresponding to the one-hot row
            if round(mat[c][r].real, 4) == 1:
                ret.append(c)
                break
    return ret


def show_cycles(qc, min_size=1, max_size=100):
    """
    Display latex of the permutation cycles of a QuantumCircuit that makes a permutation matrix
    Displays nothing if it is not a permutation matrix
    :param qc: QuantumCircuit or unitary matrix (not verified)
    :param min_size: does not display cycles less than min_size
    :param max_size: truncates cycles > max_size with ...
    :return: Display Latex - no return value
    """
    if isinstance(qc, QuantumCircuit):
        unitary = what_is_the_matrix(qc)
    else:
        unitary = qc
    cycles = []
    priors = []
    perm = permutation_integers(unitary)

    for k in range(len(perm)):
        step = k
        new_cycle = [k]

        # Handle one step cycles
        if perm[step] == k:
            cycles.append(new_cycle)
            priors.append(step)
        elif k not in priors:  # skip values already in a cycle
            # loop through until a repeat is found
            while not perm[step] == k:
                step = perm[step]
                if step in priors:
                    break
                new_cycle.append(step)
                priors.append(step)
            if len(new_cycle) > 1:
                cycles.append(new_cycle)

    latex = r'\begin{equation*}'

    for cycle in cycles:
        cycle_len = len(cycle)
        if cycle_len >= min_size:
            for step in range(len(cycle)):
                if step < max_size//2 or step > (cycle_len - max_size//2):
                    latex += str(cycle[step])
                    if step < len(cycle) - 1:
                        latex += ' \mapsto '
                elif step == max_size//2:
                        latex += r' \ldots \ldots '

            latex += r'\;\;\;({})'.format(cycle_len) + r'\\'
            # add extra linefeed -- easier to read
            latex += r'\end{equation*}' + '\r\n' + r'\begin{equation*}'

    latex += r'\end{equation*}'
    display(Latex(latex))


def get_bracket_type(bracket_type=None):
    if bracket_type is None:
        return default_bracket_type
    return bracket_type


def show_array(np_array, bracket_type=None, factor_out=True,
               normalize=False, label=None, begin_equation=True,
               display_exp=False, positive_exp=True):
    display(Latex(np_array_to_latex(np_array, bracket_type=bracket_type, factor_out=factor_out,
                                    normalize=normalize, label=label, begin_equation=begin_equation,
                                    display_exp=display_exp, positive_exp=positive_exp)))


def np_array_to_latex(np_array, bracket_type=None, factor_out=True,
                      normalize=False, label=None, begin_equation=True,
                      display_exp=False, display_omega=False, omega_size=0, positive_exp=True):
    rows, cols = np_array.shape
    bracket_type = get_bracket_type(bracket_type)

    # is omega_size is not passed in, compute it
    # would be passed in for truncated array
    if display_omega:
        if omega_size == 0:
            omega_size = np_array.shape[0]
        normalize = True
    else:
        omega_size = 0

    # Normalize forces the first term to be 1
    if normalize:
        factor = np_array[0][0]
        # only divide by real
        factor = round(factor.real, 10)
        if factor == 0:
            factor = 1
            factor_out = False
        else:
            factor_out = True
    else:
        if factor_out:
            factor = _factor_array(np_array)
            if factor == 0:
                factor_out = False
    output = ''
    if begin_equation:
        output = r'\begin{equation*}'
    if label is not None:
        output += label + ' = '
    if factor_out:
        output += format_float_as_latex(factor)
    output += r'\begin{' + bracket_type + r'matrix}'
    for i in range(rows):
        for j in range(cols):
            current = np_array[i, j]
            if factor_out:
                current = current / factor
            output += format_complex_as_latex(
                current, display_exp=display_exp, omega_size=omega_size,
                positive_exp=positive_exp)
            if j < cols - 1:
                output += ' & '
        output += r' \\ ' + '\n'
    output += r'\end{' + bracket_type + r'matrix}'
    if begin_equation:
        output += r'\end{equation*}'
    return output


def _factor_array(np_array):
    factor = 0

    rows, cols = np_array.shape
    for i in range(rows):
        for j in range(cols):
            potential = abs(round(np_array[i, j].real, 10))
            if potential != 0 and factor != 0 and potential != factor:
                return 0
            else:
                if factor == 0 and potential != 0:
                    factor = potential

            potential = abs(round(np_array[i, j].imag, 10))
            if potential != 0 and factor != 0 and potential != factor:
                return 0
            else:
                if factor == 0 and potential != 0:
                    factor = potential
    if factor == 1:
        return 0
    return factor


def format_complex_as_exponent(complex_to_format, positive_exp=True):
    # if it is just 1, don't put it into exponent
    if round(complex_to_format.real, 4) == 1:
        return format_complex_as_latex(complex_to_format, display_exp=False)

    exponent = sympy_log(complex_to_format)
    # if not pure imaginary, don't format as exponent
    if not round(float(re(exponent)), 4) == 0:
        return format_complex_as_latex(complex_to_format, display_exp=False)

    # if it can't be converted, just return the raw value
    latex = format_rotation_latex(float(im(exponent)))
    if latex == str(float(im(exponent))):
        return format_complex_as_latex(complex_to_format, display_exp=False)

    return r'e^{' + format_rotation_latex(float(im(exponent)), positive_only=positive_exp) + ' i}'


def format_complex_as_omega(complex_to_format, omega_size):
    # if it is just 1, don't format
    if round(complex_to_format.real, 8) == 1:
        return format_complex_as_latex(complex_to_format, display_exp=False)

    exponent = sympy_log(complex_to_format)
    # if not pure imaginary, don't format
    if not round(float(re(exponent)), 8) == 0:
        return format_complex_as_latex(complex_to_format, display_exp=False)

    rotation_in_radians = float(im(exponent))

    fraction = get_rotation_fraction(rotation_in_radians, positive_only=True)

    if np.isclose(np.cos(fraction * pi), np.cos(rotation_in_radians)):
        omega_val = ((omega_size // fraction.denominator) * fraction.numerator ) // 2
        return r'\omega^{'+str(omega_val) + r'}'

    return format_complex_as_latex(complex_to_format, omega_size=0)


def format_complex_as_latex(complex_to_format, display_exp=False, positive_exp=True, omega_size=0):
    if omega_size > 0:
        return format_complex_as_omega(complex_to_format, omega_size)

    if display_exp:
        return format_complex_as_exponent(complex_to_format, positive_exp=positive_exp)

    latex = ''
    if np.isclose(complex_to_format.real, 0):
        if np.isclose(complex_to_format.imag, 0):
            return ' 0 '
        else:
            if complex_to_format.imag < 0:
                latex += '-'
            if np.isclose(np.abs(complex_to_format.imag), 1):
                latex += 'i'
            else:
                latex += format_float_as_latex(np.abs(complex_to_format.imag)) + 'i'
    else:
        latex += format_float_as_latex(complex_to_format.real)
        if np.isclose(complex_to_format.imag, 0):
            return latex
        if complex_to_format.imag > 0:
            latex += '+'
        else:
            latex += '-'
        if np.isclose(np.abs(complex_to_format.imag), 1):
            latex += 'i'
        else:
            latex += format_float_as_latex(np.abs(complex_to_format.imag)) + 'i'
    return latex


def format_float_as_latex(float_to_format, max_denominator=64):
    if float_to_format < 0:
        sign = '-'
    else:
        sign = ''

    positive = np.abs(float_to_format)

    f = frac.Fraction(positive).limit_denominator(max_denominator)
    if f.denominator == 1:
        return format_raw(float_to_format)

    if np.isclose(f.numerator / f.denominator, positive):
        return sign + r'\frac{' + str(f.numerator) + '}{' + str(f.denominator) + '}'

    # handle square roots of fractions
    square = positive ** 2
    f = frac.Fraction(square).limit_denominator(max_denominator ** 2)
    # only format smaller integer fractions
    if f.numerator <= max_denominator or f.denominator <= max_denominator:
        if np.isclose(f.numerator / f.denominator, square):
            return sign + r'\frac{' + latex_sqrt(reduce_int_sqrt(f.numerator)) + '}{' + latex_sqrt(
                reduce_int_sqrt(f.denominator)) + '}'

    return format_raw(float_to_format)


def latex_sqrt(reduce):
    factor = reduce[0]
    radical = reduce[1]
    if radical == 1:
        return str(factor)
    if factor == 1:
        return r'\sqrt{' + str(radical) + '}'
    return str(factor) + r'\sqrt{' + str(radical) + '}'


def format_raw(raw):
    output = np.format_float_positional(raw, precision=4, trim='-')
    # doesn't seem to trim properly
    if output[-1] == '.':
        output = output[:-1]
    return output


def prime_factors(n):
    i = 2
    factors = []
    while i ** 2 <= n:
        if n % i:
            i += 1
        else:
            n = n / i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def reduce_int_sqrt(n):
    factor = 1
    radical = 1
    for prime, prime_group in groupby(prime_factors(n)):
        prime_exponent = len(list(prime_group))
        factor = factor * prime ** (prime_exponent // 2)
        radical = radical * prime ** (prime_exponent % 2)
    return factor, radical


def reverse_string(string):
    return string[::-1]


def int_to_binary_string(number, size, reverse=False):
    binary_string = '{0:b}'.format(number)
    binary_string = binary_string.rjust(size, '0')
    if reverse:
        return binary_string[::-1]
    return binary_string


def format_state_vector(state_vector, show_zeros=False, reverse=True):
    binary_vector = {}

    bits = int(log(len(state_vector), 2))
    for n in range(len(state_vector)):
        if show_zeros or round(state_vector[n].real, 4) != 0 or round(state_vector[n].imag, 4) != 0:
            ket_string = int_to_binary_string(n, bits, reverse=reverse)
            binary_vector[ket_string] = np.round(state_vector[n], 8)
    return binary_vector


def print_state_vector(qc, show_zeros=False, integer=False, show_prob=False, reverse=True, split=0):
    state_vector = execute_state_vector(qc)
    print_state_array(state_vector, show_zeros=show_zeros, show_prob=show_prob,
                      integer=integer, reverse=reverse, split=split)


def print_state_array(state_vector, show_zeros=False, integer=False,
                      show_prob=False, reverse=True, split=0):
    ket_format = format_state_vector(state_vector, show_zeros=show_zeros, reverse=reverse)

    for k, v in sorted(ket_format.items()):
        if not show_zeros and round(v.real, 8) == 0 and round(v.imag, 8) == 0:
            continue
        prob = ''
        if show_prob:
            prob = '(p={})'.format( np.round((v*np.conj(v)).real, 4))
        if integer:
            if split == 0:
                print('{} {}|{}>'.format(prob, v, str(int(k, 2))))
            else:
                count = len(k)
                print('{} {}|{}>|{}>'.format(prob, v, str(int(k[0:split], 2)), str(int(k[split:count], 2))))
        else:
            print(prob, v, '|', k)


def _get_array_factor(ket_format_array):
    amplitudes = []
    for k, v in (ket_format_array.items()):
        amplitudes.append(v)
    return _factor_array(np.array(amplitudes).reshape(len(amplitudes), 1))


def _format_kets(binary_string, split_array, split_color, integer):
    if len(split_array) <= 1:
        if integer:
            val = str(int(binary_string, 2))
        else:
            val = binary_string
        return r' \vert' + r'\textbf{' + val + '}' + r'\rangle '

    kets = ''
    start_at = 0
    for k in range(len(split_array)):
        if split_color is not None:
            kets += r'\color{' + str(split_color[k]) + r'}{'
        if integer:
            val = str(int(binary_string[start_at:split_array[k]], 2))
        else:
            val = binary_string[start_at:split_array[k]]
        kets += r' \vert' + r'\textbf{' + val + '}' + r'\rangle '
        start_at = split_array[k]

        if split_color is not None:
            kets += r'}'
    return kets


def _get_factored_prefix(n_complex):
    if n_complex.real < 0:
        return '-'
    else:
        if n_complex.imag < 0:
            return '-i'
        else:
            return ''


def get_bloch_vectors(qc):
    if isinstance(qc, QuantumCircuit):
        rho = what_is_the_density_matrix(qc)
    else:
        rho = qc

    bit_size = int(log2(rho.shape[0]))

    bloch_array = []
    for current_bit in range(bit_size):
        x_component = np.real(np.trace(Pauli.pauli_single(bit_size, current_bit, 'X').to_matrix() @ rho))
        y_component = np.real(np.trace(Pauli.pauli_single(bit_size, current_bit, 'Y').to_matrix() @ rho))
        z_component = np.real(np.trace(Pauli.pauli_single(bit_size, current_bit, 'Z').to_matrix() @ rho))
        bloch_array.append([x_component, y_component, z_component])
    return bloch_array


def get_bloch_angles(qc):
    if isinstance(qc, QuantumCircuit):
        bloch_array = get_bloch_vectors(qc)
    else:
        bloch_array = qc

    bloch_angles = []

    for bloch_vector in bloch_array:
        x_component, y_component, z_component = bloch_vector
        x_component = round(x_component, 14)
        y_component = round(y_component, 14)
        z_component = round(z_component, 14)

        r = sqrt(x_component**2 + y_component**2 + z_component**2)

        phi = acos(z_component/r)

        if x_component == 0:
            theta = 0
        else:
            theta = atan(y_component / x_component)

        bloch_angles.append([r, theta, phi])
    return bloch_angles


def show_bloch_angles(qc, label='\psi', global_phase=True):
    bloch_array = get_bloch_angles(qc)

    latex_bloch_vector = ''
    current_bit = 0
    for bloch_angles in bloch_array:
        theta = bloch_angles[0]
        phi = bloch_angles[1]

        latex_bloch_vector += format_bloch_vector(round(theta, 12), round(phi, 12),
                                                  label + '_' + str(current_bit), global_phase=global_phase) + llf

        current_bit += 1

    display(Latex(latex_bloch_vector))



def format_bloch_vector(theta, phi, label='\psi', global_phase=True):
    l_theta = format_rotation_latex(theta/2)
    if global_phase:
        l_phi = format_rotation_latex(phi)

        str_bloch_vector = r'\begin{equation*} \vert ' + label + r'\rangle='
        str_bloch_vector += r'cos\left({}\right)'.format(l_theta)
        str_bloch_vector += r'\vert 0 \rangle +'
        if not phi == 0:
            str_bloch_vector += r'e^{' + l_phi + r' i}'

        str_bloch_vector += r' sin\left({}\right)'.format(l_theta)
        str_bloch_vector += r'\vert 1 \rangle'
        str_bloch_vector += r'\end{equation*}'
    else:
        l_phi = format_rotation_latex(phi/2)

        str_bloch_vector = r'\begin{equation*} \vert ' + label + r'\rangle='
        if not phi == 0:
            str_bloch_vector += r'e^{-' + l_phi + r' i}'
        str_bloch_vector += r'cos\left({}\right)'.format(l_theta)
        str_bloch_vector += r'\vert 0 \rangle +'
        if not phi == 0:
            str_bloch_vector += r'e^{' + l_phi + r' i}'

        str_bloch_vector += r' sin\left({}\right)'.format(l_theta)
        str_bloch_vector += r'\vert 1 \rangle'
        str_bloch_vector += r'\end{equation*}'

    return str_bloch_vector


def show_state_vector(qc, show_zeros=False, integer=False, split=0, split_registers=False,
                      split_color=None, factor_out=True, label='\psi', truncate=128,
                      highlight=-1, display_exp=False, normalize=False):
    str_state_vector = r'\begin{equation*} \vert ' + label + r'\rangle='
    ket_format = format_state_vector(execute_state_vector(qc), show_zeros)
    is_first = True
    is_factored = False

    if factor_out:
        front_factor = _get_array_factor(ket_format)
        if front_factor > 0:
            is_factored = True
            str_state_vector += format_complex_as_latex(front_factor) + r'\big('

    if normalize and not is_factored:
        front_factor = ket_format[next(iter(ket_format))]
        if not front_factor == 1:
            is_factored = True
            str_state_vector += format_complex_as_latex(front_factor) + r'\big('

    item_count = 0
    is_highlighted = False
    vector_length = len(ket_format)
    truncate_printed = False
    if len(ket_format) > truncate:
        truncate_start = truncate // 2
        truncate_stop = vector_length - truncate // 2
    else:
        truncate_start = vector_length + 1
        truncate_stop = truncate_start + 1

    # use first value to get size
    split_array = get_split_array(qc, split, split_registers)

    for k, v in sorted(ket_format.items()):
        item_count += 1
        if item_count < truncate_start or item_count > truncate_stop:

            if highlight > 0 and item_count % highlight == 0:
                str_state_vector += r'\color{red}{'
                is_highlighted = True
            if not is_first:
                if round(v.real, 8) > 0:
                    str_state_vector += '+'
                elif round(v.real, 8) == 0 and round(v.imag, 8) >= 0:
                    str_state_vector += '+'
                elif round(v.real, 8) == 0 and round(v.imag, 8) == 0:
                    # for when show_zeros
                    str_state_vector += '+'

            is_first = False

            kets = _format_kets(k, split_array=split_array, split_color=split_color, integer=integer)

            if is_factored:
                if round(np.real(v / front_factor), 6) == 1:
                    str_state_vector += kets
                else:
                    str_state_vector += format_complex_as_latex(v / front_factor, display_exp=display_exp) + kets
            else:
                str_state_vector += format_complex_as_latex(v, display_exp=display_exp) + kets
            if is_highlighted:
                str_state_vector += r'}'
                is_highlighted = False
            # iPython breaks with equations too long.
            if item_count % 10 == 0:
                str_state_vector += r'\end{equation*}' + '\n' + r'\begin{equation*} \quad\quad\quad '
        else:
            if not truncate_printed:
                str_state_vector += r'\end{equation*} \begin{equation*} ' \
                                    r'....... \end{equation*} \begin{equation*} ' \
                                    r'\quad\quad\quad '
                truncate_printed = True
    if is_factored:
        str_state_vector += r'\big)'
    str_state_vector += r'\end{equation*}'
    display(Latex(str_state_vector))


def get_split_array(circuit, split_value, split_registers):
    full_size = 0
    for k in range(len(circuit.qregs)):
        full_size += len(circuit.qregs[k])

    if split_value > 0:
        return [split_value, full_size]
    if not split_registers:
        return [0]

    ar = []
    current_end = 0
    for k in range(len(circuit.qregs)):
        reg_len = len(circuit.qregs[k])
        current_end += reg_len
        ar.append(current_end)
    return ar


def print_short_state_vector(qc):
    ket_format = format_state_vector(execute_state_vector(qc))
    for k, v in ket_format.items():
        if v.imag != 0:
            print('{0}+{1}I |{2}> '.format(v.real, v.imag, k))
        else:
            print('{0}|{1}> '.format(v.real, k))


def decompose_single(unitary_matrix):
    (theta, phi, lamb) = twoq.euler_angles_1q(unitary_matrix)
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    qc.append(rrz_gate(lamb), [qr[0]])
    qc.ry(theta, qr[0])
    qc.append(rrz_gate(phi), [qr[0]])
    new = what_is_the_matrix(qc)
    alpha = get_global_phase(unitary_matrix, new)
    print('alpha= {}, beta= {}, gamma= {}, delta={}'
          .format(format_rotation(alpha),
                  format_rotation(phi),
                  format_rotation(theta),
                  format_rotation(lamb)))


def decompose_single_qiskit(unitary_matrix):
    (theta, phi, lamb) = twoq.euler_angles_1q(unitary_matrix)
    qc = QuantumCircuit(1)
    qc.u3(theta, phi, lamb, 0)
    new = what_is_the_matrix(qc)
    alpha = get_global_phase(unitary_matrix, new)
    print('theta= {}, phi= {}, lambda= {}, phase={}'
          .format(format_rotation(theta),
                  format_rotation(phi),
                  format_rotation(lamb),
                  format_rotation(alpha)))


def get_global_phase(original, new):
    if np.allclose(original, new):
        alpha = 0
    else:
        m_factor = original @ np.linalg.inv(new)
        if not np.isclose(m_factor[0, 0], 0):
            factor = phase(m_factor[0, 0])
        else:
            factor = phase(m_factor[0, 1])

        if np.allclose(original,
                       (np.exp(imag * factor)) * new):
            alpha = factor
        else:
            raise ValueError('New Matrix not equal to old ')
    return alpha


def decompose_single_all(decompose, fraction=8):
    found = False
    i = complex(0, 1)
    for a in range(1, 2 * fraction):
        for b in range(0, 2 * fraction):
            for c in range(0, 2 * fraction):
                for d in range(0, 2 * fraction):

                    alpha = pi - (pi / fraction) * a
                    beta = pi - (pi / fraction) * b
                    gamma = pi - (pi / fraction) * c
                    delta = pi - (pi / fraction) * d

                    ar = np.array([[np.cos(alpha) + i * np.sin(alpha), 0],
                                   [0, np.cos(alpha) + i * np.sin(alpha)]])
                    br = np.array([[np.cos(beta / 2) - i * np.sin(beta / 2), 0],
                                   [0, np.cos(beta / 2) + i * np.sin(beta / 2)]])
                    cr = np.array([[np.cos(gamma / 2), -np.sin(gamma / 2)],
                                   [np.sin(gamma / 2), np.cos(gamma / 2)]])
                    dr = np.array([[np.cos(delta / 2) - i * np.sin(delta / 2), 0],
                                   [0, np.cos(delta / 2) + i * np.sin(delta / 2)]])

                    if np.allclose(dr @ cr @ br @ ar, decompose):
                        print('alpha= {}, beta= {} gamma= {} delta= {}'
                              .format(format_rotation(alpha),
                                      format_rotation(beta),
                                      format_rotation(gamma),
                                      format_rotation(delta)))
                        found = True
    if not found:
        print('Didnt find it')


def decompose_single_u3_all(decompose, fraction=8):
    found = False
    i = complex(0, 1)
    for t in range(1, 2 * fraction):
        for l in range(0, 2 * fraction):
            for p in range(0, 2 * fraction):

                theta = pi - (pi / fraction) * t
                lam = pi - (pi / fraction) * l
                phi = pi - (pi / fraction) * p

                u = np.array([[np.cos(theta / 2), -np.exp(i * lam) * np.sin(theta / 2)],
                              [np.exp(i * phi) * np.sin(theta / 2), np.exp(i * lam + i * phi) * np.cos(theta / 2)]])

                if np.allclose(u, decompose):
                    print('theta= {}, phi= {}, lambda= {}'
                          .format(format_rotation(theta),
                                  format_rotation(phi),
                                  format_rotation(lam)))
                    found = True
    if not found:
        print('Didnt find it')


def decompose_single_qiskit_raw(unitary_matrix):
    alpha = phase(np.linalg.det(unitary_matrix) ** (-1.0 / 2.0))

    (theta, lamb, phi) = twoq.euler_angles_1q(unitary_matrix)
    return alpha, theta, lamb, phi


def execute_state_vector(qc):
    backend = Aer.get_backend('statevector_simulator')
    results = execute(qc, backend=backend).result()
    return results.get_statevector(qc)


def execute_unitary(qc):
    backend = Aer.get_backend('unitary_simulator')
    results = execute(qc, backend=backend).result()
    return results.get_unitary(qc)


def execute_real(qc, str_backend, shots):
    backend = IBMQ.get_backend(str_backend)
    job = execute(qc, backend=backend, shots=shots)
    job_monitor(job)
    results = job.result()
    return results.get_counts()


def execute_seeded(qc, shots):
    return execute_simulated(qc, shots, 12345)  # just a number that will always be the same


def execute_simulated(qc, shots, seed_simulator=None):
    backend = Aer.get_backend("qasm_simulator")
    results = execute(qc, backend=backend, shots=shots, seed_simulator=seed_simulator).result()
    return results.get_counts()


# Custom Gates

def global_gate(alpha):
    name = 'G \n(' + format_rotation(alpha) + ')'
    sub_global = QuantumCircuit(1, name=name)
    sub_global.rz(alpha, 0)
    sub_global.y(0)
    sub_global.rz(alpha, 0)
    sub_global.y(0)
    return sub_global.to_instruction()


def rrz_gate(beta):
    name = 'RRz \n(' + format_rotation(beta) + ')'
    sub_rrz = QuantumCircuit(1, name=name)
    sub_rrz.rz(beta / 2, 0)
    sub_rrz.x(0)
    sub_rrz.rz(-beta / 2, 0)
    sub_rrz.x(0)
    return sub_rrz.to_instruction()


def get_rotation_fraction(rotation_in_radians, positive_only=False):
    rotation_in_radians = rotation_in_radians % (2 * np.pi)
    if positive_only and rotation_in_radians < 0:
        rotation_in_radians = 2 * np.pi + rotation_in_radians

    return frac.Fraction(rotation_in_radians / np.pi).limit_denominator(512)


def format_rotation(rotation_in_radians, positive_only=False):
    fraction = get_rotation_fraction(rotation_in_radians, positive_only=positive_only)

    if np.isclose(np.cos(fraction * pi), np.cos(rotation_in_radians)):
        if fraction < 0:
            sign = '-'
        else:
            sign = ''
        ret = str(abs(fraction))
        ret = ret.replace('1/', 'pi/')
        if ret == '1':
            return sign + 'pi'
        if ret == '2':
            return sign + '2pi'
        return sign + ret
    else:
        return str(rotation_in_radians)


def format_rotation_latex(rotation_in_radians, positive_only=False):
    fraction = get_rotation_fraction(rotation_in_radians, positive_only=positive_only)

    num = fraction.numerator
    den = fraction.denominator

    if np.isclose(np.cos(fraction * pi), np.cos(rotation_in_radians)):
        if fraction < 0:
            sign = '-'
            num = abs(num)
        else:
            sign = ''

        if num == 0:
            return r'0'

        if den == 1:
            if num == 1:
                return sign + r'\pi'
            else:
                return sign + r'%s\pi' % num

        if num == 1:
            return sign + r'\frac{\pi}{%s}' % den
        return sign + r'\frac{%s\pi}{%s}' % (num, den)
    else:
        return str(rotation_in_radians)


def ints_to_continued_fraction(numerator, denominator):
    quotients = []
    while denominator != 0:
        quotients.append(numerator // denominator)
        # Use the integer divide and flip method
        numerator, denominator = denominator, numerator % denominator
    return quotients


def convergent_of_fraction(numerator, denominator, n):
    quotients = ints_to_continued_fraction(numerator, denominator)
    if n > len(quotients):
        n = len(quotients)
    if n < 2:
        # should not be called with depth < 2
        # but return an approximation
        return quotients[n], 1 + quotients[n + 1]

    p_0 = 1
    p_1 = quotients[0]

    q_0 = 0
    q_1 = 1
    for k in range(1, n):
        p_2 = quotients[k] * p_1 + p_0
        p_0 = p_1
        p_1 = p_2

        q_2 = quotients[k] * q_1 + q_0
        q_0 = q_1
        q_1 = q_2

    return p_2, q_2


def latex_recurse_cfraction(quotients, count, shrink_at=99):
    if count == len(quotients) - 1:
        return str(quotients[count])
    if count > shrink_at:
        frac_type = r'\; + \; \frac{1}{'
    else:
        frac_type = r'\; + \; \cfrac{1}{'
    return str(quotients[count]) + frac_type + latex_recurse_cfraction(quotients, count + 1, shrink_at) + '}'


def latex_continued_fraction(numerator, denominator, shrink_at=99):
    quotients = ints_to_continued_fraction(numerator, denominator)
    output = r'\cfrac{' + str(numerator) + '}' + '{' + str(denominator) + '} \; = \; '
    if quotients[0] > 0:
        output = output + str(quotients[0]) + '+'
    output = output + r'\cfrac{1}{' + latex_recurse_cfraction(quotients, 1, shrink_at) + '}'
    return '$' + output + '$'


def format_plot_data(answers, tick_threshold=0, spacing=8, reverse=True, integer=True, bin_size=1):
    first_key = next(iter(answers))
    bit_size = len(first_key)

    # load data
    #x_axis_data = np.arange(0, 2 ** bit_size)
    y_axis_data = [0]* (2 ** bit_size)

    for k, v in answers.items():
        key_value = int(k, 2)
        y_axis_data[key_value] = v

    # put a tick mark in the center no matter what
    tick_marks = [(2 ** bit_size // 2)]

    # put first tick mark on first one with data
    last_tick_mark = -spacing - 1

    # tick on top 10
    sorted_values = np.sort(y_axis_data)[-10::]
    for k in range(len(y_axis_data)):
        if y_axis_data[k] >= sorted_values[0] and (k - last_tick_mark) > tick_threshold:
            tick_marks = np.append(tick_marks, k)
            last_tick_mark = k

    # apply bins
    x = []
    y = []

    for k in range(0, len(y_axis_data), bin_size):
        # use the lowest edge of the bin
        bin_total = np.sum(y_axis_data[k:k + bin_size])
        if bin_total > -1:
            x.append(k)
            y.append(np.sum(y_axis_data[k:k + bin_size]))

    return x, y, tick_marks


def plot_results(answers, tick_threshold=0, fig_size=(10, 5),
                 reverse=True, integer=True, fontsize=14, spacing=8, bin_size=1):
    x_axis_data, y_axis_data, tick_marks \
        = format_plot_data(answers,
                           tick_threshold=tick_threshold, reverse=reverse,
                           integer=integer, spacing=spacing, bin_size=bin_size)

    fig, axes = plt.subplots(1, 1, figsize=fig_size)

    # rotate the binary strings so they display vertically
    if integer:
        axes.set_xticklabels(tick_marks, fontsize=fontsize)
    else:
        axes.set_xticklabels(tick_marks, fontsize=fontsize, rotation=70)

    plt.bar(x_axis_data, y_axis_data, width=4)
    plt.xticks(tick_marks)
    plt.show()


def factor_int(n):
    step = lambda x: 1 + (x << 2) - ((x >> 1) << 1)
    maxq = int(floor(sqrt(n)))
    d = 1
    q = 2 if n % 2 == 0 else 3
    while q <= maxq and n % q != 0:
        q = step(d)
        d += 1
    return [q] + factor_int(n // q) if q <= maxq else [n]


def test_period(a, period, nilf):
    a = int(a)
    period = int(period)

    t1 = pow(a, period, nilf)
    t2 = pow(a, 2 * period, nilf)

    if t1 == t2:
        return period
    return -1


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def mod_div(m, n, nilf):
    (g, a, b) = egcd(m, nilf)
    if not g == 1:
        return -1

    return a * n % nilf


def binary_powers(nilf, base=2):
    ar = []
    max_power = int(log(nilf, 2) + 1)
    for k in range(max_power + 1):
        val = pow(base, 2 ** k, nilf)
        ar.append(val)
    return ar


def get_powers(val, nilf, base=2):
    size = int(log(val, 2) + 2)
    ar = binary_powers(nilf, base)
    ret = 'print (( '
    for k in range(size):
        pos = 2 ** k
        if pos & val > 0:
            ret = ret + str(pow(base, pos, nilf)) + ' * '
    ret = ret[:-2]
    return ret + ') % nilf )'


def inverse_powers(nilf):
    ar = []
    inv = mod_inverse(2, nilf)
    max_power = int(log(nilf, 2) + 1)
    for k in range(max_power + 1):
        val = pow(inv, 2 ** k, nilf)
        ar.append(val)
    return ar


def get_inverse_powers(val, nilf):
    size = int(log(val, 2) + 2)
    ar = inverse_powers(nilf)
    inv = mod_inverse(2, nilf)
    ret = '( '
    for k in range(size):
        pos = 2 ** k
        if pos & val > 0:
            ret = ret + str(pow(inv, pos, nilf)) + ' * '
    ret = ret[:-2]
    return ret + ') % nilf'


def int_prod(ar):
    # numpy has 64 bit limit, so use this
    ret = 1
    for x in ar:
        ret *= x
    return ret


def find_period(nilf, base=2):
    ar_factors = []
    a, b = primefactors(nilf)
    factors = factorint((a - 1) * (b - 1))
    for f, v in factors.items():
        for k in range(v):
            ar_factors.append(f)

    for k in range(len(ar_factors)):
        old = ar_factors[k]
        ar_factors[k] = 1
        if not pow(base, int_prod(ar_factors), nilf) == 1:
            ar_factors[k] = old
    return int_prod(ar_factors)


def get_nilf(a, b):
    nilf = prime(a) * prime(b)
    period = find_period(nilf)
    return nilf, period


def nilf_stat(nilf):
    """
    Prints information about a Number I'd Like to Factor (nilf).
    :param nilf: Number I'd Like to Factor
    :return: True/False whether it can be factored by period finding in base 2.
    """
    bit_size = len('{0:b}'.format(nilf)) + 1
    factors = primefactors(nilf)

    period_factors = primefactors((factors[0] - 1) * (factors[1] - 1))
    print(nilf)
    print('bit_size:', bit_size, 'factors:', factors, 'period_factors:', period_factors)
    print('Full factors:', factorint((factors[0] - 1) * (factors[1] - 1)))

    period = find_period(nilf, base=2)

    p = pow(2, period // 2, nilf)
    g1 = gcd(p - 1, nilf)
    g2 = gcd(p + 1, nilf)

    shors = False
    # only even periods are considered
    if period // 2 == period / 2:
        if (g1 > 1 and g1 < nilf) or (g2 > 1 and g2 < nilf):
            shors = True
    print('Shors base 2:', shors, ', Period:', period)
    return shors


