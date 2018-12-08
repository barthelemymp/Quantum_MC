# Project description

The Quantum Monte Carlo algorithm allows the computation of observables 
on a chain of quantum spins. Quantum spins are modelled using the XXZ 
model. A further description is given in the articles assaad_evertz_world_line.pdf.

# Directory structure

*ExactComputation
  * contains script allowing the exact computation of the energy for a given chain up to eight spins.
  * contains a script testloop.py allowing the computation using the loop algorithm to verify it. 
  * contains files .txt with the results of those computations

* structure_code
  * contains the localclass and the loopclass, two different Quantum Monte Carlo algorithms
  * check_loop_ergodicity.py verifies the ergodicity of the loop algorithm
  * show_loop_algorithm.ipynb contains an animation on the loop algorithm
  *config contains the parameter of a physical system, the monte carlo is there.
  *compute local is the code that made the data using local montecarlo

*oldcodes
   *former code, when we tried a local and periodic montecarlo

* report
  * contains the project report
  * contains the figures and the .tex file

* results
  *VaryM
    *contains scripts than compute the file .txt storing the result of long computation, compute_x=z=1_beta1.py and compute_x=1z=05_beta1.py 
    *contains the file .txt QMCENERG
    *contains the scripts than show the graphs ThvsComp
  *loop_result.ipynb computes the autocorrelation graph
  *local_result contains graphs made with results of montecarlo using local loves

* documentation
  * contains documentation used along the project