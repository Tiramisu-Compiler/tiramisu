This is the k-best Model, where the LSTM recursive model predicts the k (here, k=5) best loop interchanges to apply on the Tiramisu function in hand, in the process of auto-scheduling. 
In the past version, the model's output can predicts interchanges up to depth = 15. The program representation is similar to the cost model's.
In the current version, the program representation has been changed (Please check the draft of the paper). We only consider 7 loops. Also, the fusion transformation has been added.
-L.M
