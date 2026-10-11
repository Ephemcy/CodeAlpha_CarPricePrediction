# Task 3: Car Price Prediction with Machine Learning

## Objective
Predict used-car selling prices from vehicle features.

## Data preparation
The supplied `car data.csv` has 301 rows and 9 columns. `Selling_Price` is the target. Predictors include car name, year, present price, kilometres driven, fuel type, selling type, transmission, and owner count. Numeric missing values are imputed with the median, categorical missing values with the most frequent value, and categorical fields are one-hot encoded.

## Model
A Random Forest Regressor was trained using an 80/20 hold-out split (`random_state=42`). Preprocessing is fitted inside a pipeline to reduce data leakage.

## Evaluation
- **MAE:** 0.605 price units
- **RMSE:** 0.921 price units
- **R²:** 0.963

The plot `actual_vs_predicted.png` compares observed and predicted prices. Lower MAE/RMSE is better; an R² closer to 1 indicates stronger fit on the test set.

## Conclusion
On this test split, the model captured about 96.3% of the variation in selling price. Performance can differ for new market conditions or other regions.

## Run
Install dependencies using `pip install -r requirements.txt`, then run `python car_price_prediction.py`.

## Limitations
The dataset is relatively small and does not specify currency in its columns. The model should be validated on current, representative market data before being used for actual pricing.
