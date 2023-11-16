# Install
see instructions in howto_install directory.

# How to reproduce
The scripts for reproduction are in the scripts directory and sorted by example. The resulting data used for plots and tables in the paper are stored in the data directory (and the corresponding subfolders).

By default the scripts for reproduction use the MKL pardiso solver which we recommend to use as well if available. Otherwise you may have to swap "pardiso" with "umfpack" in some of the files (search->replace).

## 4.1.2 Vector Laplacian on sphere:
* Go to scripts/4_1_2_VecLap_sphere and
* make an empty directory data_sphere
  ```mkdir data_sphere``` 
* execute the file with arguments specifying polynomial and geometry order, e.g.
  ```python3 sphere_problem --ku 1 --kg 1```
generates the files for geometry order 1 and polynomial degree 1.
* The generated files are in data_sphere and are:
  * comparison_nze_...dat: comparison of non-zero entries
  * comparison_ndof_...dat: comparison of number of dofs
  * comparison_ncdof_...dat: comparison of number of globally coupled (i.e.  after condensation) dofs
  * comparison_L2Error_tang_...dat: comparison of L2 error in the tangential plane
  * comparison_L2Error_normal_...dat: comparison of L2 error in normal direction
  * comparison_H1Error_...dat: comparison of H1 error
  
## 4.1.3 Vector Laplacian on house of cards:
* Go to scripts/4_1_3_VecLap_house_of_cards and
* make an empty directory data_kink
  ```mkdir data_kink```
* execute the file with arguments specifying polynomial and the height of the house of cards, e.g.
  ```python3 kink_problem.py --ku 3 --H 0```
  ```python3 kink_problem.py --ku 3 --H 0.8660254037844386```
generates the files for height 0 and sqrt(3/4). 
* The generated files are in data_kink and are:
  * comparison_nze_...dat: comparison of non-zero entries
  * comparison_ndof_...dat: comparison of number of dofs
  * comparison_ncdof_...dat: comparison of number of globally coupled (i.e. after condensation) dofs
  * comparison_L2Error_tang_...dat: comparison of L2 error in the tangential plane
  * comparison_L2Error_normal_...dat: comparison of L2 error in normal direction
  * comparison_H1Error_...dat: comparison of H1 error
  
## 4.2 Stokes
* Go to scripts/4_2_Stokes and
* make an empty directory data_stokes
```mkdir data_stokes```
* execute the file "convergence.py" with arguments specifying polynomial orders, the viscosity nu and the number of mesh refinements (and additionally the number of threads for a shared parallelization with --nthreads)
   ```python3 convergence.py --ku 2 kg 4 --nref 3 --nu 1```
* This generates files of the corresponding errors with respect to an uniform mesh refinement in the directory data_stokes
* execute the file "convergence_nu.py" with arguments specifying polynomial orders and the number of levels i such that the viscosity is given by 10**(-i)
   ```python3 convergence_nu.py --ku 2 kg 4 --nref 8 ```
* This generates files of the corresponding errors with respect to a decreasing viscosity in the directory data_stokes

## 4.3 Generalization of the Schäfer Turek benchmark for surfaces
Go to scripts/4_3_Schaefer_Turek and
* make an empty directory data:
  ```mkdir data```
* execute 
```python3 schaefer_turek.py --mapping="roled" --ku 4 --kg 5 --maxh 0.05 --dt 0.001```
Here the mapping can be:
   * "flat": flat geometry
   * "flat2": flat geometry but mesh line at x=1.1 (kink mesh)
   * "kink": kinked geom (Phi1)
   * "roled": isometric curved mapping (Phi2)
   * "friesA": non-isometric curved mapping from Fries (Phi3)
   * "friesB": non-isometric curved mapping from Fries (does not appear in the paper)
The other parameters correspond to polynomial order, geometry order, maximum mesh size parameter and time step size.

* This will generate vtk output in the current directory. and:
  * schaefer_turek_xxx.dat: some integral quantities and the pressure difference (first line labels the columns)
  * schaefer_turek_xxx.txt: some computational number (dofs, etc.)

## 4.4 Kelvin-Helmholtz
* Go to scripts/4_4_Kelvin_Helmholtz. Edit KelvinHelmholtz.py and set the following variable for the different cases:
  * Gamma_0 (section 4.4.1): set case = 0, bonus_intorder = 8, order = 8, hmax = 0.05
  * Gamma_1 (section 4.4.1): set case = 1, bonus_intorder = 0, order = 8, hmax = 0.05
  * Gamma_2 (section 4.4.1): set case = 2, bonus_intorder = 0, order = 8, hmax = 0.05
  * Gamma_3 (section 4.4.1): set case = 3, bonus_intorder = 0, order = 8, hmax = 0.05
  * sphere (section 4.4.2): set case = 4, bonus_intorder = 5, order = 5, hmax = 0.07
* Execute
```python3 KelvinHelmholtz.py```
* This will generate output in the directory xxx_data where xxx corresponds to your hostname. The output files are vtk files (one per time unit) and:
  * Re_1000_P_8_bio_8_h_0.1dt_3.571428571428572e-05.dat: time series of divergence L2 error, kinetic energy, palinstrophy, enstrophy (first row labels the columns)
  * Re_1000_P_8_bio_8_h_0.1dt_3.571428571428572e-05.txt: for computational quantities (number of elements, dofs, etc..)

## 4.5 Stanford bunny
* Go to scripts/4_5_Stanford_Bunny and execute
  ```python3 bunny.py```
* This will generate a directory xxx_data (xxx being the hostname) with vtk files (one for every time unit) and:
  * P_4_bio_4_h_dt_0.005.txt: for computational quantities
  * P_4_bio_4_h_dt_0.005.data: for integral quantities (as for KH example)
