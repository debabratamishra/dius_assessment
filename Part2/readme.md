## Cost Optimization using AWS


### Infrastructure based optimizations

<b> Reviewing current infrastructure</b> : <br>
Reviewing the current EC2 instance types that are being used is necessary for analysing and optimizing the cost vs performance tradeoff, e.g they may be using more powerful compute instances than needed. Moving to cheaper instance families like m5 or m6 could help reduce costs as per the suitability of the model.

<b>Serverless Inference</b> : <br>
Serverless options like AWS Lambda might prove to be an effective solution. This removes infrastructure management and scales automatically, optimizing costs for bursty workloads. The computer vision model can be deployed as a serverless function, triggered by events such as frame processing requests.

<b>Scalability vs Efficiency</b> : <br>
For scalability, using GPU during realtime inferencing of the computer vision model would turn out to be more efficient than a CPU. Using GPU instances like P2, P3 or G3 will reduce the compute time required during serving.

<b>Auto Scaling</b> : <br>
The EC2 autoscaling policies should be configured properly to match workload needs. Having too much buffer capacity running idle will waste money.

<b>Asynchronous Inference</b> : <br>
If the video payloads are considerably large, using SageMaker asynchronous endpoints, which allow for processing of large input payloads without the need for real-time inference would be effective. This can be more cost-effective for burst traffic and large payloads.

I found the following article, particularly useful.
https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/

### Model optimizations

<b>GPU vs CPU Instances</b> : <br>
Transitioning the workload from CPU-based EC2 instances to GPU instances would prove to be an effective solution for Machine learning solutions working at a scale. GPUs are highly efficient for training and inferencing deep learning models and can significantly accelerate inference time. This would result in processing the same amount of data in much lesser time and thereby saving the compute time.

<b>Postprocessing optimization</b> : <br>
Post-processing model outputs via Lambda and pushing the aggregated results to DynamoDB instead of processing every frame is more efficient. It would be cost effective than storing all video frames.

A rough architecture of the possible pipeline is shown below
![Alt text](AWS-Cost-Optimization.png?raw=true "AWS based cost optimization pipeline")