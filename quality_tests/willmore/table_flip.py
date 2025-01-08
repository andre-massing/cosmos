# %%

import pandas as pd
import numpy as np

def flip_rows_and_columns(file_path, output_path=None):
    """
    Reads a .dat file with headers (first row and column), flips rows and columns (transposes),
    and saves/prints the result.
    
    Args:
        file_path (str): Path to the .dat file to read.
        output_path (str, optional): Path to save the transposed data. If None, prints to console.
    """
    try:
        # Load the data into a pandas DataFrame, assuming tab-delimited by default
        df = pd.read_csv(file_path, sep="\t", index_col=0)
        
        # Transpose the DataFrame
        flipped_df = df.transpose()

        vertical_labels =  flipped_df.axes[0].to_list()
        horizontal_labels = ["dh-dt"] + flipped_df.axes[1].to_list()
        output = np.column_stack((vertical_labels, flipped_df.values))

        if output_path:
            df = pd.DataFrame(output, columns=horizontal_labels)
            df.to_csv(output_path, sep='\t', index=False)
            print(f"Transposed data saved to {output_path}")
        else:
            # Print the transposed DataFrame
            print("Transposed data:")
            print(flipped_df)
    
    except Exception as e:
        print(f"An error occurred: {e}")

tests_names = ['./circle_tests/', './half_sphere_tests/', './sphere_tests/',
               './sc_half_sphere_tests/', './sc_sphere_tests/', './torus_tests/', './half_torus_tests/', './moving_sphere_tests/']

for test_name in tests_names:
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1, 2]:

                flip_rows_and_columns(test_name + str(i) + str(j) + str(k) + '/displacement.dat',
                        test_name + str(i) + str(j) + str(k) + '/displacement_flipped.dat')
                flip_rows_and_columns(test_name + str(i) + str(j) + str(k) + '/mean.dat',
                        test_name + str(i) + str(j) + str(k) + '/mean_flipped.dat')

# %%
