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
1. Run MapleStory and make sure the game is on windowed mode and game window size is resized to smallest
2. Turn on minimap on the top-left corner of the game window
3. Execute the script and press 'F2' to take a screenshot
4. Edit saved screenshot and crop your character nametag and use it to replace name_tag.png

## Config
Please check config.py before running, customized to your keyboard settings

## Run
Run command
```
python mapleStoryAutoLevelUp.py --map <name_of_the_map> --monsters <name_of_the_monsters> --attack <skill>
```
Exmaple for north forest training ground 2(北部森林訓練場2)
```
python mapleStoryAutoLevelUp.py --map north_forest_training_ground_2 --monsters green_mushroom,spike_mushroom --attack magic_claw
```
Exmaple for fire land 2(火焰之地2)
```
python mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack magic_claw
```
Example for ant_cave_2(螞蟻洞2)
```
python mapleStoryAutoLevelUp.py --map ant_cave_2 --monsters spike_mushroom,zombie_mushroom --attack aoe_skill
```
Example for cloud_balcony(雲彩露臺)
```
python mapleStoryAutoLevelUp.py --map cloud_balcony --monsters brown_windup_bear,pink_windup_bear
```
```
Example for lost_time_1(遺失的時間1)
python mapleStoryAutoLevelUp.py --map lost_time_1 --monsters evolved_ghost --attack aoe_skill
```
Note that after script execution, you need to click back to Maple Story game window (Make sure the game windows is your active window)

You can press 'F1' to pasue/continue the script control
You can press 'F2' to take a screenshot, which will save to scrennshot/

## Supported Map 
1. north forest training ground 2(北部森林訓練場2)
2. fire land 2(火焰之地2)
3. ant_cave_2(螞蟻洞2)
4. cloud_balcony(雲彩露臺)
5. lost_time_1(遺失的時間1)
6. north forest training ground 8(北部森林訓練場8)
7. monkey_swamp_3(猴子沼澤地3)
8. first barrack (第一軍營)

## Supported Monsters
1. fire pig(火肥肥)
2. green mushroom(綠菇菇)
3. spike mushroom(刺菇菇)
4. zombie mushroom(殭屍菇菇)
5. black axe stump(黑斧木妖)
6. brown windup bear(褐色發條熊)
7. pink windup bear(粉色發條熊)
8. evolved ghost(進化妖魔)
9. wind single eye beast(風獨眼獸)
10. angel monkey(天使猴)
11. skeleton soldier(骷髏士兵)
12. skeleton officer(骷髏隊長)

If you want to try this script on other map/monster, you need to add new map to minimaps/ and add monsters icon to monster/

## Route recorder
To make the route design easier, here is a new script that listen to player's keyboard and record
the player input on route map.

To invoke the route recorder, please use the following command:
```
python routeRecorder.py --new_map <map_directory_name>

Example:
python routeRecorder.py --new_map my_new_map
```
while running this recoder
Press 'F1' to stop/resume the recoder
Press 'F2' to take screenshot
Press 'F3' to save current route map and start record a new one
Press "F4" to save current map

## Legacy Version
This project previously use full-size screenshot map for camera localization
and route planning. However, I found that capturing player location from top-left corner minimap in the game is easier and more reliable.

Therefore, I developed a new localization scheme based on minimap, and all the previously maps/ are migrated to minimaps/ to benefit from this change. If you still want to use the old camera/player localization method. Please use the following command:

```
python mapleStoryAutoLevelUp_legacy.py --map <name_of_the_map> --monsters <name_of_the_monsters> --attack <skill>

Exmaple:
python mapleStoryAutoLevelUp_legacy.py --map lost_time_1 --monsters evolved_ghost --attack aoe_skill
```





