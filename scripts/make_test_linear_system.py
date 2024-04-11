"""This module generates test code for the validate_solvers() function in tests.rs."""

pyperclip_imported = False

try:
    import pyperclip
    pyperclip_imported = True
except ModuleNotFoundError:
    pyperclip_imported = False


def main():
    n = 20
    SOLUTION_VECTOR = [2*i for i in range(n)]
    DIAGONAL_COEFFS = [i+1 for i in range(n)]
    FIELD_SIZE = 6
    MATRIX_NAME = 'a'
    VECTOR_NAME = 'b'
    source_term = []
    pretty_system = ""
    matrix_definition_code = f"let mut {MATRIX_NAME}_coo: CooMatrix<Float> = CooMatrix::new({n}, {n});\n"
    vector_definition_code = f"let {VECTOR_NAME} = DVector::from_column_slice(&["
    for i in range(n):
        source_term_value = 0
        pretty_system += "| "
        for j in range(n):
            value = 0
            if i == j:
                value = DIAGONAL_COEFFS[i]
            elif abs(i - j) == 1:
                value = -DIAGONAL_COEFFS[i] / 4
            source_term_value += SOLUTION_VECTOR[j] * value
            pretty_system += f"{value: >{FIELD_SIZE}}"
            if value != 0:
                if isinstance(value, int):
                    value = str(value) + '.'
                matrix_definition_code += f"{MATRIX_NAME}_coo.push({i}, {j}, {value});\n"
        source_term.append(source_term_value)
        pretty_system += f" | {'=' if i==(n-1)/2 else ' '} | {source_term_value: >3} |\n"
        vector_definition_code += f"{source_term_value}"
        if i < n-1:
            vector_definition_code += ", "

    matrix_definition_code += f"\nlet {MATRIX_NAME} = CsrMatrix::from(&a_coo);"
    vector_definition_code += "]);"

    print(pretty_system)
    print(matrix_definition_code)
    print(vector_definition_code)

    pretty_system = '\n'.join(["// " + line for line in pretty_system.split('\n')])

    if pyperclip_imported:
        pyperclip.copy(pretty_system + '\n\n' + matrix_definition_code + '\n' + vector_definition_code)
        print("Code copied to clipboard")


if __name__ == "__main__":
    main()
