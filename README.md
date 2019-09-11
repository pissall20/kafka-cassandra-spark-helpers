# iot_poc
Project Indigo

#### Step 1: Install Cassandra
In Ubuntu: [Cassandra Installation link](https://cassandra.apache.org/doc/latest/getting_started/installing.html)

In MacOS: `brew install cassandra`

#### Step 2: Configure settings
Have a look at `settings.py` in the root folder and set the relevant settings for development/production

#### Step 3: Set up the database
You can run `setup.py` and it will create a `key-space` and `table` for you. You might have to edit `create_random_data()` in the same file, to suit your schema from the settings. It will run fine with the default settings.
