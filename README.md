# Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields
The goal of this research is to study aquatic environments such as lakes, rivers, and bayous by gathering information on spatiotemporal variability. Traditionally, we rely on manual sampling by humans from such environments or sparse measurements from static sensors. In such scenarios, we envision to utilize an underwater robot
with multi-modal sensors to gather in-situ measurements. However, it is also crucial to reason about both spatial and temporal variations of water currents for efficient robotic information gathering. Therefore, a team of students in their capstone project will compute informative robot trajectories under predicted water flow dynamics that will drive the robot with motion constraints to several locations for gathering the maximum information from an aquatic environment. In this project,
students will train a deep transformer neural network for learning water flow dynamics in continuous domains from a large real water current dataset. They will then develop a prototype software tool for implementing the proposed sampling-based information gathering algorithm that will generate robot trajectories for random samples on predicted water flow dynamics. Finally, students will validate the overall approach with extensive simulation runs and multiple physical experiments with a real robot. Through this project, students will gain experience in applying deep learning models, robot motion planning algorithms, experimental validation, and performance analysis along with handling the hardware and software of a physical underwater robot.

--------------------------------

**Predicting future data values using a Transformer Model:**

![thumbnail_Salt](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/7ca93f91-5d33-4d95-bebc-f900508f2b7f)

![Velocity](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/6260bdf8-c909-4872-981b-ca0e31fef673)

--------------------------------

**Creating robot trajectories based on predictions:**

![Screenshot_2024-03-30_at_7 33 20_PM](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/666f9f16-8dab-445c-9dfb-9d9c18dfe82d)

--------------------------------

(In Progress) Creating an algorithm to make assumptions for data values not traversed by the robot:


-------------------------------
**Instructions**

**DataLoaderCSV.py**
Before getting into this, I want to summarize how our data is split among the files briefly. Each .nc file has 900 rows of data when laid out in a .csv file, each representing data taken at a different location. There is a file for data taken every 6 hours spanning 11 months. In total this leaves us with 1337 files.
1) First you need to create 4 folders of .nc files
   1) Entire_Input
   2) Entire_Target
   3) Testing_Input
   4) Testing_Target

      You will take 5 chronological files and put them in Entire_Input and put the following 6th file in the Entire_Target. In my case, I manually repeated this process until I had 1000 files in Entire_Input and 200 files in Entire_Target
      This is because my model looks at the first 5 files and tries to predict the 6th file's data values. We will need several sets like this to properly train with.
      For example, files (1-5) [6] (7-11) [12] (13-17) [18]...
      () = Entire_Input
      [] = Entire_Target

      For Testing_Input and Testing_Target you will only do a single set. So 5 chronological files will go in Testing_Input and the 6th file will go in the Testing_Target. This is so we can test our model later on.

**NewReorganize.py**
This will reorganize the Entire_Input.csv file for training. Above I mentioned that it looks at the first 5 files and predicts the 6th file. However, it is a bit more complicated than this. 
In each file there are 900 rows of data. My model will look at the first 5 rows in Entire_Input and predict the 1st row in Entire_Target, then look at rows 6-10 in Entire_Input and predict the 2nd row in Entire_Target. In order to "look at the first 5 files and predict the 6th file" we need it to do the following:

look at first row of file 1 (row 1 Entire_Input)
look at first row of file 2 (row 2 Entire_Input)
look at first row of file 3 (row 3 Entire_Input)
look at first row of file 4 (row 4 Entire_Input)
look at first row of file 5 (row 5 Entire_Input)
predict the first row of file 6 (row 1 Entire_Target)

look at second row of file 1 (row 6 Entire_Input)
look at second row of file 2 (row 7 Entire_Input)
look at second row of file 3 (row 8 Entire_Input)
look at second row of file 4 (row 9 Entire_Input)
look at second row of file 5 (row 10 Entire_Input)
predict the second row of file 6 (row 2 Entire_Target)

... so on until it does this 900 times
After doing this we will be left with all 900 rows of the predictions for the 6th file in that sequence trained from 5 files (4500 rows).
While training it will do this all over again with new data while training if you give it more sequences.
Note that ONLY Entire_Input and Testing_Input needs to be reorganized

**Transformer.py**
This file trains the model as briefly described above AND runs a test with the testing folders which creates a predictions.csv
The predictions.csv is then used to compare the predictions and actual values in Temperature.py, Salt.py, and Velocity.py

Here you will use the 2 .csv files as shown. This will train the model with inputs and targets.
![image](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/1228c0d3-18a3-4b38-baa0-c083a05915fc)

Here you will put the New_Testing_Input.csv file as shown and this test will output a predictions.csv
![image](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/80a9a898-f05f-4e3b-9ca3-348e02f8e375)

**Temperature.py, Salt.py, Velocity.py**
All of these files work similarly. 

At the top, you will need to include the predictions.csv that you have from running the model file and the Testing_Target.csv so we can compare them to the actual values.
![image](https://github.com/Isaiahensley/Investigation-of-Informative-Robot-Trajectory-Planning-on-Predictive-Vector-Fields/assets/143129356/18d2478f-76e1-4811-b0d0-8ed9d9ec5a46)
