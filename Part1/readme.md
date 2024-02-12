# Challenge 1
## Metrics derived from the model
Training performance - <br>
Validation MAE : 4.895 

Performance on unseen data - <br>
MAE :  7.675 <br>
MSE :  89.431 <br>
RMSE : 9.456 <br>

## Instructions
1. Clone the repository to local machine
2. Build Docker image using the command `docker build -t dius-regression .`
3. Once the process is completed, run the container using the following command <br>
`docker-compose up`
4. The service should be up and running at http://0.0.0.0:8178. Use POST calls to tes the service. A sample curl call is mentioned below - <br>
`curl --json '{"X1": "14.59355382431799","X2": "41.18622465124143","X3": "33.9978936605386","X4": "64.4769023500858","X5": "108.81641054516794","X6" : "79.18062928342908", "X7" : "70.75284656795137", "X8" : "109.25455637899452", "X9" : "123.59616528814512", "X10" : "QWE"}' http://0.0.0.0:8178/predict`

## If you had flexibility to choose a model to tackle this problem, what model would that be?

Given a flexibility to choose a model, I would choose a decision tree based algorithm such as XGBoost. It would be a better choice than choosing a neural network based architecture given the relatively smaller training data size(9000 rows) with few predictors(10). Due to this, with complex model architecture there's a higher likelihood of model overfitting, especially if not properly regularized or tuned (e.g. a basic XGBoost regression model without any hyperparameter tuning gives a mean absolute error of ~10.0 on the unseen test dataset).

Using hyperparameter tuning and feature engineering on the dataset, a model like XGBoost would produce better performance than a neural network. But, with an increase in the training data size and the number of predictors, a complex neural network would be a better choice.