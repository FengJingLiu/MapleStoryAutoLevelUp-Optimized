# Maple Story Auto Level Up

An auto leveling up script for Maple Story Artale

![Intro Animation](media/intro.gif)

[▶ Watch demo on YouTube](https://www.youtube.com/watch?v=QeEXLHO8KN4)

This work purely-based on Computer Vision technique, it doesn't required access game's memory. Instead, it detects image pattern(i.e., player nametag and monsters) on game window screen and send simulated keyboard command to the game to control player's character.

## Environment
Windows11

Python3.12

OpenCV4.11

## Install dependency
```
pip install -r requirements.txt
```

## Preparation
Please replace nameTag.png to your own MapleStory character nametag. Including the badge in nametag.png can increase player localization stability

## Config
Please check config.py before running, customized to your keyboard settings

## Run
### Step1: Run Maple Story Artale
### Step2: Move character to the map you want to level up
### Step3: Make sure the game is on windowed mode and game window size is resized to smallest 
### Step4: Execute mapleStoryAutoLevelUp.py
```
Run execute command:
python mapleStoryAutoLevelUp.py --map <name_of_the_map> --monsters <name_of_the_monsters>

Exmaple command for north forst training ground 2(北部森林訓練場2)
python mapleStoryAutoLevelUp.py --map north_forst_training_ground_2 --monsters green_mushroom,spike_mushroom

Exmaple command for fire land 2(火焰之地2)
python mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump

Example command for ant_cave_2(螞蟻洞2)
python mapleStoryAutoLevelUp.py --map ant_cave_2 --monsters spike_mushroom,zombie_mushroom

Example command for cloud_balcony(雲彩露臺)
python mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear

Example command for lost_time_1(遺失的時間1)
python mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost
```

### Step5: Click back to Maple Story game window (Make sure the game windows is your active window)

## Supported Map 
1. north forst training ground 2(北部森林訓練場2)
2. fire land 2(火焰之地2)
3. ant_cave_2(螞蟻洞2)
4. cloud_balcony(雲彩露臺)
5. lost_time_1(遺失的時間1)

## Supported Monsters
1. fire pig(火肥肥)
2. green mushroom(綠菇菇)
3. spike mushroom(刺菇菇)
4. zombie mushroom(殭屍菇菇)
5. black axe stump(黑斧木妖)
6. brown windup bear(褐色發條熊)
7. pink windup bear(粉色發條熊)
8. evolved ghost(進化妖魔)

If you want to try this script on other map/monster, you need to add new map to maps/ and add monsters icon to monster/
