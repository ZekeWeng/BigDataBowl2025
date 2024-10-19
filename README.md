# BigDataBowl2025



## Instructions for Start

```
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Prepare data file
python preprocessing.py
```

### General Plan
1) Prediction for Man vs Zone (Percentages)
2) Prediction for Run vs Pass (Percentages)
3) Matchup Calculation
4) Presnap Favorability
   i) P(Run)(P(Man)Advantage + P(Zone)Advantage)+ P(Pass)(P(Man)Advantage + P(Zone)Advantage)
5) Derivative Metrics


### Actionables Oct 12
1) Data preprocessing (Zeke)
   a) Merging player_play into full data
   b) Add some of those columns into visualizations
   c) Are there obvious patterns where offense / defense is advantages? Matching Offense + Defense

2) Distance (Victor)
   a) How does distance traveled by offense AND/OR defense affect yards gained? (Split this up for run and pass plays)
      CAVEAT - Set then motion (not just going to the line of scrimmage)
      i) Offensive distance traveled =>
      ii) Defensive distance traveled =>
      iii) Some Formula for mixing them =>

3) Does motion help offense at all? (Justin + Cayden)
   a) Label types of motion + type of player in motion
   b) Players + Positions + Type of Motion Frequencies
   c) How do these labels affect play result? (yards gained)


### Actionables Oct 19
1) Data preprocessing (Zeke)
   a) Relook at the playDirection for yards gained
   b) Visualization
   c) Feature Engineering => Man/Zone

2) Feature
   a) Feature Engineering => Run/Pass

3) Does motion help offense at all? (Justin + Cayden)
   a) Label types of motion + type of player in motion
   b) Players + Positions + Type of Motion Frequencies
   c) How do these labels affect play result? (yards gained)
