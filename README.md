# Predicting-Plant-Traits

A pretrained Resnet50 convolutional model was used to extract 2048 key photographic features of plant photographs, captured through citizen science. Pairing this with 163 ancillary features allowed for a rich feature set, which was then pipelined into a tuned boosting algorithm (XGBoost) to accurately predict target features of unseen images. Ultimately, this resulted in an average R² test score of approximately 0.3 across the six target features, showcasing successful model performance.

See report.pdf for a full breakdown of the project.
