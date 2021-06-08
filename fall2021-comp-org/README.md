# Introduction to Computer Architecture & Organization (EECE3026)
Offered by Dr. Phillip Wilsey.
Detail about the class can be seen in [Coursicle](https://www.coursicle.com/uc/courses/EECE/3026/) and [Dr. Wilsey's website](https://eecs.ceas.uc.edu/~wilseypa/).

## Projects
There are four projects to be graded as assignments.
Projects 1 and 2 are to practice gate level circuit simulation software.
They are of little value to showcase.

Projects 3 and 4 were accomplished by a team of [Anthony Gamerman](mailto:gamermad@mail.uc.edu) and Liu (Zuguang Liu).

### Project 3: Non-Pipelined Control Unit
A single-bus processing unit was designed capable of executing a specified instruction set.
The design was implemented at the gate level.
See [Project 3 detail document](project-3-handout.pdf).

Our solution includes a [design document](project-3-report.pdf), as well as a working gate-level simulation.

The simulation can be opened and executed in [Logisim Evolution](https://github.com/logisim-evolution/logisim-evolution), an actively maintained fork of Logisim.
Compatibility with original Logisim is not sure and not tested.
Please note that `project-3-main.circ` uses `project-3-control-unit.circ` as library, and they need to be in the same directory.

### Project 4: Pipelined Control Unit
The required control unit in this project has similar specification with Project 3, except the control flow and data flow can be pipelined with multiple stages.
See project [Project 4 detail document](project-4-handout.pdf).

As the solution does not require a gate-level implementation, we simply shipped a [report document](project-4-report.pdf) that describes the detail of our design.