### Table 4: Artificial Bee Colony Optimization Configuration

| Parameter                   |   Configured Value | Description                                               |
|:----------------------------|-------------------:|:----------------------------------------------------------|
| Number of Food Sources (SN) |                 10 | Population of candidate solutions                         |
| Employed Bees               |                 10 | Assigned 1:1 to food sources for neighborhood exploration |
| Onlooker Bees               |                 10 | Select food sources probabilistically via roulette wheel  |
| Abandonment Limit (limit)   |                  5 | Max consecutive unimproved trials before scout reset      |
| Maximum Cycles              |                 10 | Total iterations of ABC search loop                       |
| Maximum Fitness Evaluations |                240 | Hard computational budget limit                           |
| Random Seed                 |                 42 | Ensures reproducible stochastic search operations         |
