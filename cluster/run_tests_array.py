import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python array_test.py <number>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Please provide an integer number.")
        sys.exit(1)

    # Decide what to do based on the input number
    if n == 1:
        import test_bachini_random_2D
    elif n == 2:
        import test_bachini_random_2D_duanli
    elif n == 3:
        import test_bachini_random_3D
    elif n == 4:
        import test_bachini_random_3D_duanli
    elif n == 5:
        import test_bachini_tanh_2D
    elif n == 6:
        import test_bachini_tanh_2D_duanli
    elif n == 7:
        import test_bachini_tanh_3D
    elif n == 8:
        import test_bachini_tanh_3D_duanli
    elif n == 9:
        import test_helfrich_blood_cell_441
    elif n == 10:
        import test_helfrich_blood_cell_441_duanli
    elif n == 11:
        import test_helfrich_blood_cell_551
    elif n == 12:
        import test_helfrich_blood_cell_551_duanli

if __name__ == "__main__":
    main()