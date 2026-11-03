### Table 1: AI4I 2020 Dataset Summary and Feature Decisions

| Feature                 | Original Type          | Usage Decision    | Rationale                                                             |
|:------------------------|:-----------------------|:------------------|:----------------------------------------------------------------------|
| UDI                     | Integer                | Dropped           | Arbitrary sequential index; zero predictive utility                   |
| Product ID              | Categorical            | Dropped           | Product serial code (redundant with Type; high cardinality)           |
| Type                    | Categorical            | Retained (OHE)    | Product quality variant (Low, Medium, High) affecting failure risk    |
| Air temperature [K]     | Numerical              | Retained          | Ambient operating temperature recorded by sensors                     |
| Process temperature [K] | Numerical              | Retained          | Direct machinery operational process temperature                      |
| Rotational speed [rpm]  | Numerical              | Retained          | Spindle rotational velocity in revolutions per minute                 |
| Torque [Nm]             | Numerical              | Retained          | Motor torque delivered to the cutting head                            |
| Tool wear [min]         | Numerical              | Retained          | Cumulative cutting tool operating time in minutes                     |
| Machine failure         | Binary                 | Target Variable   | Primary classification target (0 = Normal, 1 = Failure)               |
| TWF                     | Binary                 | Dropped (Leakage) | Tool Wear Failure indicator; directly reveals machine failure         |
| HDF                     | Binary                 | Dropped (Leakage) | Heat Dissipation Failure indicator; directly reveals machine failure  |
| PWF                     | Binary                 | Dropped (Leakage) | Power Failure indicator; directly reveals machine failure             |
| OSF                     | Binary                 | Dropped (Leakage) | Overstrain Failure indicator; directly reveals machine failure        |
| RNF                     | Binary                 | Dropped (Leakage) | Random Failure indicator; directly reveals machine failure            |
| temperature_difference  | Numerical (Engineered) | Candidate Feature | Process Temp - Air Temp; captures machine thermal dissipation deficit |
