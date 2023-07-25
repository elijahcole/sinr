## Web App for Visualizing Model Predictions

Gradio app for exploring different model predictions.

####  Downloading the pretrained models
To use the web app, you must first download the pretrained models from [here](https://data.caltech.edu/records/dk5g7-rhq64/files/pretrained_models.zip?download=1) and place them at `sinr/pretrained_models`. See `app.py` for the expected paths. 

####  Starting the app
Activate the SINR environment:
```bash
 conda activate sinr_icml
```
Navigate to the web_app directory:
```bash
 cd /path/to/sinr/web_app
```
Launch the app:
```bash
 python app.py
```
Click on or copy the local address output in the command line and open this in your web browser in order to view the web app. This will look something like:
```bash
 Running on local URL:  http://127.0.0.1:7860
```
#### Controlling the app
* From here use your mouse and the dropdown menus to choose which model and species you wish to visualize.
* Taxon IDs are aligned with those from [iNaturalist](iNaturalist.org), so if you wish to find a specific taxon you can search within the iNaturalist site and then copy the taxon ID into the web app. Note that not all taxa from iNaturalist are present in all models. 
    * For example, to view the predicted species range for the Northern Cardinal, navigate to the iNaturalist page for this taxon (https://www.inaturalist.org/taxa/9083-Cardinalis-cardinalis) and set the taxon ID in the app to `9083` and click "Run Model".
* To generate a thresholded predicted range select the "threshold" button and use the slider to choose the threshold value.
