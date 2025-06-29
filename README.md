# Maple Story Auto Level Up

An auto leveling up script for Maple Story Artale

<img src="media/intro.gif" width="60%">

[▶ Watch demo on YouTube](https://www.youtube.com/watch?v=QeEXLHO8KN4)

This work purely-based on Computer Vision technique, it doesn't required access game's memory. Instead, it detects image pattern(i.e., player red health bar and monsters) on game window screen and send simulated keyboard command to the game to control player's character.

✅ No memory access required

✅ Purely screen-based CV detection

✅ Simulates real keyboard input

## Environment
* Windows11
* Python3.12
* OpenCV4.11

Note: this project DOES NOT support virtual environment(VM), it's only for recreational and academical use.

## Support MapleStory Version
This project is mostly developed and tested on MapleStory Artale Taiwan Server

It also supports global server, use '--cfg global' to load customized global config 

## Install dependency
```
pip install -r requirements.txt
```

## Preparation
1. Run MapleStory and make sure the game is on windowed mode and game window size is resized to smallest
2. Turn on minimap on the top-left corner of the game window
3. Create a party in the game(press 'P' and click 'build'), and make sure a red bar shows on top of your character
4. Set up your own key-binding at config/config_edit_me.yaml

## Run
Run command
```
python mapleStoryAutoLevelUp.py --map <map_name> --monsters <monster1,monster2,...> --attack <attack_mode>
```
Exmaple for north forest training ground 2(北部森林訓練場2)
```
python mapleStoryAutoLevelUp.py --map north_forest_training_ground_2 --monsters green_mushroom,spike_mushroom --attack directional
```
Exmaple for fire land 2(火焰之地2)
```
python mapleStoryAutoLevelUp.py --map fire_land_2 --monsters fire_pig,black_axe_stump --attack directional
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

Press 'F1' to pasue/continue the script control

Press 'F2' to take a screenshot, which will be saved to scrennshot/

## Supported Map 
1. north forest training ground 2(北部森林訓練場2)
2. fire land 2(火焰之地2)
3. ant_cave_2(螞蟻洞2)
4. cloud_balcony(雲彩露臺)
5. lost_time_1(遺失的時間1)
6. north forest training ground 8(北部森林訓練場8)
7. monkey_swamp_3(猴子沼澤地3)
8. first barrack (第一軍營)
9. dragon territory (魔龍領地)
10. empty house (空屋)
11. mushroom hill (菇菇山丘)
12. pig shores (肥肥海岸)

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
13. wild kargo (魔龍)
14. pig (肥肥)
15. ribbon pig (緞帶肥肥)
16. cold eye (冰獨眼獸)


## Want to Make a New Map? → Route Recorder

To design a custom route more easily, you can use the `routeRecorder.py` script.
It listens to your keyboard inputs and records them onto a route map.

Use the following command in your terminal to start recording:

```
python routeRecorder.py --new_map <map_directory_name>
```
| Key  | Action                                     |
| ---- | ------------------------------------------ |
| `F1` | Pause or resume the recorder               |
| `F2` | Take a screenshot (saved to `screenshot/`) |
| `F3` | Save current route map and start a new one |
| `F4` | Save the current map to map.png            |

## Want to Make a New Monster? → Mob Maker

You can find the names of the monsters to be added at the following website:

[Maplestory GMS 65](https://maplestory.wiki/GMS/65/mob)

```
python mob_maker.py

>Fetching mobs from: https://maplestory.io/api/GMS/65/mob
>You can find monster names at https://maplestory.wiki/GMS/65/mob
>Enter mob name:Snail  <-- Example
```

Automatically download monster PNG images, excluding death animation frames, since monsters do not need to be attacked again after death and therefore do not require recognition.

The monster actions such as `hit`, `move`, `skill`, and `stand` are retained. While it's uncertain whether keeping so many actions will affect performance, the expectation is that having a greater variety of monster animations will enhance the diversity and accuracy of monster recognition.

Once the download is complete, you can find the downloaded image in the `Monster/{MonsterName}` folder.

## Auto Dice Roller
An light-weighted auto-bot that help you roll the dice in character creation page.

User can assign the desire attributes and let the script do the job for you

```
python AutoDiceRoller.py --attribute <STR,DEX,INT,LUK>

Example: for creating a full-INT wizard character:
python AutoDiceRoller.py --attribute 4,4,13,4
```

## Legacy Version
This project previously use full-size screenshot map for camera localization
and route planning. However, I found that capturing player location from top-left corner minimap in the game is easier and more reliable.

Therefore, I developed a new localization scheme based on minimap, and all the previously maps/ are migrated to minimaps/ to benefit from this change. If you still want to use the old camera/player localization method. Please use the following command:

```
python mapleStoryAutoLevelUp_legacy.py --map <name_of_the_map> --monsters <name_of_the_monsters> --attack <skill>

Exmaple:
python mapleStoryAutoLevelUp_legacy.py --map lost_time_1 --monsters evolved_ghost --attack aoe_skill
```

## ☕ Support the Developer

If you find this project helpful, consider supporting the developer by buying me a coffee!

> 💡 You can type in any amount you like — $1, $5, or $10 — whatever you're comfortable with.  
> 💵 Tips are in **USD**, not NTD.

[![Buy Me a Coffee](https://img.shields.io/badge/%F0%9F%92%96_Tip_me_$1_or_more-yellow?style=flat-square&logo=buymeacoffee)](https://www.buymeacoffee.com/kenyu910645)
