##----------------------------------------------------------------
##                        Load libraries                        --
##----------------------------------------------------------------
for (lib in c("tidyverse", "keras", "magrittr", "caret", "reticulate", "RSocrata", "tensorflow")) {
    lib |>
        library(character.only = TRUE) |>
        base::suppressPackageStartupMessages() |>
        base::suppressWarnings()
    # Remove lib variable
    if (base::all(c("tidyverse", "keras", "magrittr", "caret", "reticulate", "RSocrata", "tensorflow") %in%
        (base::.packages()))) {
        base::rm(lib)
    }
}


##---------------------------------------------------------------
##                       Set Python Path                       --
##---------------------------------------------------------------
###  Next we are going to walk through directories to find python.exe and set     
###  the Python executable path. This must be done because both Keras and         
###  Tensorflow are Python libraries, bound to by R at runtime. This means that   
###  those Python libraries have to be present on your machine.

python_version <- "3.8" ##  Select the Python version to use
if (reticulate::py_config()$version != python_version) {
    python_path <- base::Sys.getenv("LOCALAPPDATA") %>%
        fs::dir_ls(path = ., recurse = TRUE, type = "directory", regexp = python_version %>%
            base::gsub("\\.", "", .) %>%
            base::paste0("Python", .)) %>%
        .[1] %>%
        base::list.files(path = ., pattern = "python.exe", full.names = TRUE)
    reticulate::use_python(python = python_path)  ##  Set the Python path
}
base::rm(python_version) %>% 
	base::suppressWarnings()


##---------------------------------------------------------------
##                  Import & Reformat Dataset                  --
##---------------------------------------------------------------
api_token <- base::Sys.getenv("bunker_fuel_API_TOKEN") ##  Retrieve the API Token from the environment variable
api_PW <- base::Sys.getenv("bunker_fuel_API_PW") ##  Retrieve the API Password from the environment variable

##  Check if the API keys are present
if (base::all(base::sapply(c("api_PW", "api_token"), function(var) {
    base::is.null(get(var))
})) == TRUE) {
    stop("One of the API keys were not found in the environment variable.")
}


##  Query for Dataset
bunkerdf <- 
	RSocrata::read.socrata(
		url = base::paste0("https://agtransport.usda.gov/resource/4v3x-mj86.json?$where=year <=",
								 lubridate::today() %>%
								 	lubridate::year()), 
    app_token = api_token, 
    email = "Cedric.Moore@envivabiomass.com",
    password = api_PW
)
base::rm(api_token, api_PW) %>%
    base::suppressWarnings()


###  Next we want to start prepping the dataset to be trained. We want to   
###  remove the columns we dont need and rename them for simplicity.        
bunkerdf <- 
	bunkerdf %>%
		dplyr::select(dplyr::contains(c("day", "marine_gas_oil"))) %>%
		dplyr::rename(price = marine_gas_oil, date = day) %>% 
		dplyr::mutate(dplyr::across(dplyr::where(lubridate::is.POSIXct), as.Date)) %>% 
		dplyr::mutate(dplyr::across(dplyr::where(is.character), as.numeric))



###  When dealing with non-continuous dates, such as missing days or weekends,   
###  it can disrupt the sequential nature of the data. This can impact the       
###  model's ability to learn and generalize patterns effectively. To avoid      
###  disruption, we will fill in missing prices with the previous days price.    
bunkerdf <- 
	base::seq(
		from = bunkerdf$date %>%
	    base::min(), 
		to = bunkerdf$date %>%
	    base::max(), 
		by = "day") %>%
		base::data.frame(date = .) %>%
		dplyr::left_join(., bunkerdf, by = "date") %>%
		tidyr::fill(price, .direction = c("downup")
	)


##----------------------------------------------------------------
##          Rescale Input Data to Improve Convergence           --
##----------------------------------------------------------------
###  Next we will rescale the data using Z-score scaling to help LSTM networks   
###  converge faster during the training process. When input features have       
###  significantly different scales, it can lead to slower convergence or        
###  difficulties in finding the optimal solution. Rescaling the data to a       
###  similar scale can aid in the convergence of the network, allowing it to     
###  learn more efficiently.    
scale_input <- c(base::mean(bunkerdf$price), stats::sd(bunkerdf$price))

##  Instead of splitting the data, we will use the full dataset for training. 
scaled_price <- 
	bunkerdf %>% 
		dplyr::select(price) %>% 
		base::scale(., center = scale_input[1], scale = scale_input[2])


##---------------------------------------------------------------
##                   Create Lagged Variables                   --
##---------------------------------------------------------------
# ###  To create predictions, the LSTM algorithm takes in a sequence of input       
# ###  data points and processes them through the network's layers. The LSTM's      
# ###  internal mechanisms, such as the input gate, forget gate, and output gate,   
# ###  allow it to learn and capture relevant patterns, dependencies, and           
# ###  temporal information from the input sequence.                                
p <- 30  ##  Prediction
n <- p  ##  Number of lagged variables to include

###  In the case of predicting marine gas oil prices for the next 30 days, it   
###  would be appropriate to include 29 lagged variables so that each           
###  prediction is based on the previous 30 values.                             
train_x <- 
	base::array(data = base::as.numeric(base::unlist(base::t(base::sapply(1:(base::length(scaled_price) -
	    n - p + 1), function(x) scaled_price[x:(x + n - 1), 1])))), dim = c(base::nrow(scaled_price) -
	    n - p + 1, n, 1)
	    )

train_y <- 
	base::array(data = base::as.numeric(base::unlist(base::t(base::sapply((1 +
	    n):(base::length(scaled_price) - p + 1), function(x) scaled_price[x:(x +
	    p - 1)])))), dim = c(base::nrow(base::t(base::sapply((1 + n):(base::length(scaled_price) -
	    p + 1), function(x) scaled_price[x:(x + p - 1)]))), p, 1)
	    )

##  Prepare input data for the next 30 days
test_x <- utils::tail(bunkerdf$price, p) 

##  Scale and transform it.
x_scaled <- 
	base::scale(test_x, center = scale_input[1], scale = scale_input[2]) %>%
	  base::as.numeric() %>%
	  base::array(data = ., dim = c(1, n, 1)
	  		)	



##---------------------------------------------------------------
##                         Build Model                         --
##---------------------------------------------------------------
###  The model architecture includes two LSTM layers with 50 units each, both    
###  returning sequences. The stateful parameter is set to TRUE, indicating      
###  that the model will maintain its state between batches. Dropout layers      
###  with a dropout rate of 0.5 are added after each LSTM layer to prevent       
###  overfitting. Finally, a time-distributed dense layer with 1 unit is added   
###  using the time_distributed function.                                      
lstm_model <- keras::keras_model_sequential()
lstm_model %>%
    keras::layer_lstm(units = 50, batch_input_shape = c(1, p, 1), return_sequences = TRUE,
        stateful = TRUE) %>%
    keras::layer_dropout(rate = 0.5) %>%
    keras::layer_lstm(units = 50, return_sequences = TRUE, stateful = TRUE) %>%
    keras::layer_dropout(rate = 0.5) %>%
    keras::time_distributed(keras::layer_dense(units = 1)
    			)


##----------------------------------------------------------------
##                        Begin Training                        --
##----------------------------------------------------------------
##  We are going to use the compile function to configure the model for training. 
##  Loss function: Mean Absolute Error (MAE)
##  Optimizer: Adaptive Moment Estimation (Adam) 
##  Metrics: Accuracy metric for classification tasks
lstm_model %>%
    keras::compile(
    	loss = "mean_absolute_error",
    	optimizer = keras::optimizer_adam(),
      metrics = "accuracy"
)

###  Set early stopping to prevent overfitting and improve the efficiency of   
###  the training process                                                      
early_stopping <- 
	keras::callback_early_stopping(
		monitor = "accuracy",
		patience = 10
)

###  Train the model by iteratively adjusting the model's parameters to       
###  minimize the loss function and improve its performance on the training   
###  data.                                                                    
lstm_model %>%
    keras::fit(
    	x = train_x, 
    	y = train_y,
    	epochs = 20,
    	batch_size = 1,
    	shuffle = FALSE,
    	callbacks = early_stopping
)

##---------------------------------------------------------------
##                        Forecast Model                       --
##---------------------------------------------------------------
###  Make predictions with LSTM model (lstm_model) and extract the   
###  forecasted values.                                                 
lstm_model_forecast <- 
	lstm_model %>%
		stats::predict(train_x, batch_size = 1) %>%
		.[, , 1] %>% 
		"*"(scale_input[2]) %>% 
		"+"(scale_input[1])


