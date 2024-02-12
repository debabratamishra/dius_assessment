## Metrics derived from the model
Training performance - <br>
Validation MAE : 4.895 

Performance on unseen data - <br>
MAE :  7.675 <br>
MSE :  89.431 <br>
RMSE : 9.456 <br>

## Instructions
1. Clone the repository to local machine
2. Build Docker imagee using the command `docker build -t dius-regression`
3. Once the process is completed, run the container using the following command <br>
`docker-compose up`
4. The service should be up and running at http://0.0.0.0:8178. Use POST calls to tes the service. A sample curl call is mentioned below - <br>
`curl --json '{"X1": "14.59355382431799","X2": "41.18622465124143","X3": "33.9978936605386","X4": "64.4769023500858","X5": "108.81641054516794","X6" : "79.18062928342908", "X7" : "70.75284656795137", "X8" : "109.25455637899452", "X9" : "123.59616528814512", "X10" : "QWE"}' http://0.0.0.0:8178/predict`