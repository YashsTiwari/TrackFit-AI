# **TrackFit AI**

![TrackFit AI Excersise image 1](https://github.com/YashsTiwari/Fitness-Tracker-using-Sensors-and-ML/blob/4f1134e90e35ebb6c5a5a45684626cb86a63b7fa/extras/Excersise_1_final.png)
- Green Zone (Correct Range): Indicates ideal posture for optimal and safe form.
- Red Zone (Incorrect Range): Indicates unsafe or improper posture, with increased risk of injury.
  
**TrackFit AI** is a machine learning-driven fitness tracking system designed to detect and analyze workout movements from sensor data. This project utilizes posture detection, motion tracking, and data analysis to provide insights into exercise performance. It supports tracking of barbell exercises, including Squat, Bench Press, Deadlift, Overhead Press, and Barbell Row, with machine learning models to evaluate form and count repetitions accurately.

## **Project Overview**

**TrackFit AI** uses sensor data from a wristband or similar wearable to classify exercises, track repetitions, and evaluate form. By analyzing accelerometer and gyroscope data, the system can recognize distinct workout patterns and assess form with machine learning, enhancing the accuracy and depth of workout tracking.

![TrackFit AI Excersise image 2](https://github.com/YashsTiwari/Fitness-Tracker-using-Sensors-and-ML/blob/4f1134e90e35ebb6c5a5a45684626cb86a63b7fa/extras/Excersise_2_final.png)

### **Benefits of this Project**

This project can be integrated with wearables(any smart watch), enabling users to enhance workouts through data-driven insights. Potential future enhancements include new exercises, improved model accuracy, and real-time tracking.

### **Key Functionalities**

1. **Data Processing and Outlier Detection**  
   - **Data Cleansing**: Standardizes raw sensor data (accelerometer and gyroscope readings), removing irrelevant data.
   - **Outlier Detection**:
     - **Interquartile Range (IQR)**: Identifies and removes extreme values.
     - **Chauvenet’s Criterion**: Filters improbable values under normal distribution.
     - **Local Outlier Factor (LOF)**: A distance-based approach to flag low-density points.

2. **Feature Engineering**
   - **Signal Processing and Transformation**:
     - **Low-pass Filtering**: Reduces noise for clarity in sensor data.
     - **Principal Component Analysis (PCA)**: Reduces dimensionality, keeping essential features.
   - **Temporal and Frequency Abstractions**:
     - **Temporal Abstraction**: Captures trends over time.
     - **Frequency Abstraction**: Uses Fourier Transformation to identify frequency patterns.
   - **Feature Selection**: Extracts important features like movement angles, joint velocities, and frequencies.

3. **Predictive Modeling**
   - **Model Training and Selection**: Trains various algorithms, including Decision Trees, K-Nearest Neighbors, SVM, Neural Networks, and Random Forests.
   - **Hyperparameter Tuning**: Uses Grid Search for cross-validation and model optimization.
   - **Exercise Classification**: Achieves ~97% accuracy in identifying exercises.

4. **Repetition Counting**
   - **Repetition Detection**: Uses peak detection in signal data to count exercise repetitions.
   - **Counting Algorithm**: Detects cycles in accelerometer or gyroscope data.
   - **Real-time Counting Integration**: Designed for real-time applications.

5. **Visualization and Analysis**
   - **Exercise Visualization**: Graphs showing sensor data and filtered signals.
   - **Form Evaluation Visualization**: Shows posture analysis with correct (green) and incorrect (red) zones.
   - **Model Evaluation Metrics**: Confusion matrices, accuracy scores, precision-recall curves, etc.


## **Project Structure**

```
├── LICENSE
├── Makefile           <- Commands for automated project actions, e.g., `make data`, `make train`.
├── README.md          <- Overview and usage of the project for developers and users.
├── data
│   ├── external       <- Third-party data sources.
│   ├── interim        <- Intermediate, transformed datasets.
│   ├── processed      <- Final, model-ready datasets.
│   └── raw            <- Original, unmodified data.
│
├── models             <- Serialized models, predictions, and summaries.
│
├── references         <- Data dictionaries and reference materials.
│
├── reports            <- Analysis reports in HTML, PDF, or LaTeX.
│   └── figures        <- Generated figures for reporting.
│
├── requirements.txt   <- Python package dependencies.
│
├── setup.py           <- Makes the project pip-installable as a Python module.
├── src                <- Source code organized by functionality.
│   ├── data           <- Scripts for dataset generation and processing.
│   │   └── make_dataset.py
│   ├── features       <- Scripts to engineer and transform features.
│   │   └── build_features.py
│   ├── models         <- Scripts for model training and prediction.
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization  <- Scripts to generate visualizations.
│       └── visualize.py
```

## **Installation**

1. Clone the repository.
2. Install dependencies from `requirements.txt` or `environment.yml` (for conda environments).
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the directory structure is as specified above, with all data folders in place.

## **Usage Guide**

### Data Preparation

Run the `make_dataset.py` script to preprocess the raw sensor data:
```bash
python src/data/make_dataset.py
```

### Feature Engineering

Execute `build_features.py` to create advanced features:
```bash
python src/features/build_features.py
```

### Model Training

Use `train_model.py` to train and optimize classification models:
```bash
python src/models/train_model.py
```

### Repetition Counting

Run `count_repetitions.py` to apply filters and count repetitions:
```bash
python src/features/count_repetitions.py
```

### Visualization

Generate workout and feature visualizations with `visualize.py`:
```bash
python src/visualization/visualize.py
```

### Real-time Application

Integrating data from fitness bands enables TrackFit AI to track and correct exercise postures in real time. This feature enhances productivity by promoting correct form, which reduces the risk of injuries significantly. With AI-driven posture analysis, the need for a personal trainer is minimized, helping users save on training costs while achieving a safer and more effective workout experience.

## **Results and Analysis**

### Model Performance

The classification model achieves **~97% accuracy** in detecting exercise types, ensuring highly reliable tracking of workout patterns.

### Repetition Counting Accuracy

Repetition counting is calibrated against known repetitions, providing accurate tracking.

![TrackFit AI Results](https://github.com/YashsTiwari/Fitness-Tracker-using-Sensors-and-ML/blob/4f1134e90e35ebb6c5a5a45684626cb86a63b7fa/extras/Results.png)
- Blue Bars: Actual
- Red Bars: Predicted

## **Impact of TrackFit AI**

**TrackFit AI** has the potential to revolutionize personal fitness training by introducing machine learning insights into workouts:

- **Improves Form and Safety**: Provides real-time feedback on posture.
- **Enhances Training Efficiency**: Tracks repetitions and exercises without manual input.
- **Makes Fitness Data Actionable**: Provides insights for data-driven workout adjustments.
- **Supports Personalized Training**: Enables feedback and recommendations based on past performance.

## **Future Directions**

1. **Integration with Additional Wearables**  
   Extend support to more wearable devices like Apple HealthKit and Google Fit. Develop a Bluetooth interface for real-time data input.

2. **Enhanced Model Accuracy with Multi-sensor Data**  
   Improve accuracy by integrating data from multiple sources (e.g., heart rate, oxygen levels). Use data fusion and multi-modal learning for enhanced context.

3. **Addition of New Exercises and Activities**  
   Expand recognition to bodyweight exercises, cardio, and complex movements. Use transfer learning to generalize across a broader range of activities.

4. **Advanced Real-time Feedback and Personalized Insights**  
   Implement reinforcement learning to adjust exercise feedback based on user form, fatigue, and progress. Track user history and provide personalized recommendations.

## **Want to Contribute?**

Contributions are wholeheartedly welcomed! To get started, fork this repository, make your changes, and submit a pull request. Let’s build a better fitness tracking experience together.

## **License**

Distributed under the MIT License. See `LICENSE` for more information.
