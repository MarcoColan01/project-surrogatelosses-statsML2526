# Project title: Double Descent in Linear Models


## Evaluation: criteria and timeline
The evaluation of the project will mostly focus on the **correctness** and **completeness** of the methodology employed, rather than on the accuracy of the trained model(s) or training time. The **reproducibility** of the results and the **quality of the report** also play a substantial role in the final evaluation.
Within approximately 3 to 4 weeks after each deadline, the submissions will be reviewed and the students will be contacted to set the date of an oral examination, usually held online but possibly in person upon request, where the project will be thoroughly discussed with the TAs; students are required to prepare a brief presentation (~5/10 mins) of their work (without slides), they will be asked to motivate the choices made in the assignment and discuss the results.
We stress that group projects are not allowed: students must complete their projects individually.

## Coding and report specifications
The preferred language is **Python**, any other choice must be agreed upon with the TAs. Jupyter notebooks are allowed. All the required algorithms, unless specified otherwise, must be **implemented from scratch**, external libraries like **numpy** and **matplotlib** may be used **to only ease computation and plotting results.** The submitted code should be reproducible and run in reasonable time (see the datasets section).

The report must be written in LaTeX and the submitted repository must contain the final PDF. The report should describe the work done to complete the assignment, discussing in particular the choices made regarding the datasets, the implementation, and the visualizations. Furthermore, the report should provide a thorough analysis of the results, highlighting both the positive and the negative aspects. The report should be 5-10 pages long.

## Datasets
The assignments do not specify the exact dataset for the task, instead you are required to pick or generate datasets and motivate the associated choices in the report. **All the listed types of dataset must be included in the project, the report must include a comparison of the results across the datasets.**

For **real-world datasets,** standard sources such as the **UCI Machine Learning Repository** or **sklearn.datasets** are recommended. When selecting a dataset, make sure it is consistent with the task at hand, using classification datasets for classification problems and regression datasets for regression problems.

For **synthetic datasets,** you are **free to design the data-generating process.** A typical approach is to sample inputs from a chosen distribution (e.g., Gaussian or uniform distribution), possibly in multiple dimensions, and then define the labels as a deterministic function of the inputs. For instance, a linear function can be used to generate linearly separable data. Noise can be added to make the problem more realistic and to study robustness.

**When applicable, datasets should be split into training and test sets** and special care must be taken to ensure that **no data leakage occurs.** In these cases, the model must be trained exclusively on the training set and performance should be evaluated on the test set. If a different evaluation protocol is more appropriate for a specific assignment, it should be clearly justified.

The running time of the algorithms is not a primary concern for this project, avoid excessively large datasets and prefer small- to medium-scale problems that allow you to run multiple experiments efficiently.


## Assignment Specifications

Paper:
Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off
Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal
PNAS 2019

The course presents linear regression and generalization through the classical bias-variance trade-off, where increasing model complexity reduces bias but increases variance, leading to a U-shaped test error curve. This framework has long been a cornerstone of statistical learning theory and guides model selection and regularization.
Recent empirical findings, however, show that this picture is incomplete. In modern high-dimensional settings, especially when models are overparameterized, the test error can exhibit a second descent after the interpolation threshold, where the model perfectly fits the training data. This phenomenon, known as double descent, challenges the classical intuition.
The paper by Belkin et al. provides a systematic study of this effect in simple settings such as linear regression. It shows that increasing model complexity beyond the interpolation point can actually improve generalization. The goal of this assignment is to highlight where classical theory fails and to motivate the need for new perspectives on generalization in modern machine learning.




### Objective
- Empirically reproduce the double descent curve

### Required dataset
- Synthetic regression dataset with (noisy) linear labels
- Real-world regression dataset


### Tasks
- Fix n, vary dimension d
- Implement and train:
    - least squares
    - ridge regression
- Measure:
    - train error
    - test error



### Expected output
- Plot of test error vs model complexity
- Identification of interpolation threshold
- Comparison with ridge

### Extensions (che voglio fare entrambe)
- Gradient descent vs closed-form
- Effect of noise








