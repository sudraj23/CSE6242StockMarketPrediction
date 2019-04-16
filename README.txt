#########################################################################################################

						CSE-6242 Data and Visual Analytics Project
						Stock Price Movement using News Analytics
				
#########################################################################################################
By : Aditya Aggarwal,Anna M. Rihele, Emily T. Huskin, Manish Mehta, Ravi P. Singh, Sudhanshu R. Singh

CSE 6242 course project on analysis of stock price movements due to news sentiments.

System Requirements:
	1. Install Flask
	2. Python 3.6 and packages(xgboost, scipy, sklearn, pandas, numpy, matplotlib, sqlite3, re, json, datetime, gc)
	3. windows-10/Mac/linux 16GB RAM(if you want to retrain the model)
				
File Structure:
	- DOC
		team03report.pdf
		team03poster.pdf
	- CODE
		-app
			myproject.py
			routes.py
			- data
				market_train[1-5].csv(1.03GB)
				news_train[1-10].csv(4.4GB)
				-SQLdatabase	
					MarketData.db
					NewsData.db

			- scripts
				generate_sqlDB.py
				train_model.py

			- static
				- images
				- lib
				script.js
				style.css
				asset.json
				finalized_model.sav
				.....
			
			- Python Notebooks
				....
		
			-templates
				index.html
				main.html
	

DESCRIPTION:

	DOC: 
		Contains the final report and the poster presented for the entire project.
		
	CODE: 
		app:
			myproject.py: imports the application instance.

			routes.py: Contains all the python code to interact with the SQL database and the model upon requests in javascript/d3 code.

			data:
				1. Dataset is hosted on kaggle competition "Two Sigma: Using News to Predict Stock Movements"(https://www.kaggle.com/c/two-sigma-financial-news).
				2. The data has been downloaded and has been split into multiple csv files. The data can be downloaded from One-drive (https://gtvault-my.sharepoint.com/:f:/g/personal/aaggarwal301_gatech_edu/EkyVBfDSD-9LrZupsMSprFYBdX286Em4KxzT_-Xkpi4cKw?e=tWXuLP)
				
			scripts:
				1. train_model.py : 
					It lodas the raw data(*.csv), process it, trains the XG-Boost model and dumps the trained model for Flask app.
					The model has been pretrained and saved in static/finalized_model.sav since it takes a lot of time to train the model from the scratch.
				2. generate_sqlDB.py: 
					It reads the raw data(*.csv) files, creates the asset.json, MarketData.db and NewsData.db for Flask app to read the data and use it for interactive visualisation.
					The databases has already been created and hosted on one-drive to download, since it takes a lot of time to generate it because of the size of data.

			static
				1. lib folder contains the d3 library and the underscore library.
				2. images conatains the required images
				3. asset.json contains the name of all the stocks and is used by the d3 code to generate the list of stocks
				4. script.js contains the entire d3 code for visualization
				5. virtualscroller.js contains the code to create the two scrollers on the visualization page.
				6. style.css contains the style elements
				7. im1,im2,im3 and im4 are used by tool tip to display the comparison between model prediction and what actually happened
				
			Python Notebooks:
				1. The folder contains python notebook with intial EDA, feature engineering and triage to select the model and manually  engineer the features.

			templates:
				1. The folder contains the html files of the app(main.html, index.html). main.html displays the home page and index.html displays the d3 visualization.
				
INSTALLATION:
	Note: We have already dumped the final model for ease of execution, so you can skip these steps of retraining the model if desired.
	
	1. Download the data from the link mentioned above (downloaded from kaggle and splitted into multiple files; 1,000,000 records per file)
	2. Once the directory structure is in place, go to CODE/app/scripts and type "python train_model.py". 
	3. This runs the entire model, trains it on the given set of data as maintained above and finally dumps the model "finalized_model.sav" into the CODE/app/static directory for use in the backend of visualisation. 
	4. For visualization of the project, perform the steps mentioned under EXECUTION below.

EXECUTION:
	1. Define environment variable
		"set FLASK_APP=myproject.py" on windows 
		"export FLASK_APP=myproject.py" on mac
	2. Navigate to CODE/app folder and type: 
		"flask run"
	3. 127.0.0.1/ renders the home page (use firefox browser)
	4. 12.0.0.1/index renders the interactive visualization(use firefox browser)
	5. Navigation through the visualization is explained on the home page of the app itself.

	
	
