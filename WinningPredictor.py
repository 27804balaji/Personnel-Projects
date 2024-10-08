import pandas as p

match = p.read_csv(r"C:\Users\abc\Downloads\archive (6)\IPL - Player Performance Dataset\IPL\IPL_Matches_2008_2022.csv")
match.rename(columns = {'ID' : 'Match_Id'} , inplace = True)
# print(match.head(3))

match_data = p.read_csv(r"C:\Users\abc\Downloads\archive (6)\IPL - Player Performance Dataset\IPL\IPL_Ball_by_Ball_2008_2022.csv")
match_data.rename(columns = { 'ID' : 'Match_Id' ,
                              'innings' : 'Innings' ,
                              'overs' : 'Over' ,
                              'ballnumber' : 'Ball' ,
                              'batter' : 'Batsman' ,
                              'bowler' : 'Bowler' ,
                              'non_striker' : 'NonStriker' ,
                              'extra_type' : 'Extra_Type' ,
                              'batsman_runs' : 'Batsman_Run' ,
                              'extra_run' : 'ExtraRun' ,
                              'total_run' : 'Total_Run' ,
                              'non_boundary' : 'Non_Boundary' ,
                              'isWicketDelivery' : 'Is_Wicket_Delivery' ,
                              'player_out' : 'Player_Dismissed' ,
                              'kind'  : 'Dismissal_Kind' ,
                              'fielders_involved' : 'Fielder' ,
                              'BattingTeam ' : 'BattingTeam'} , inplace = True)

for index, match in match_data.iterrows():
    toss_winner = match.get('TossWinner')
    team1 = match.get('Team1')
    team2 = match.get('Team2')

    if toss_winner == team1:
        match_data.at[index, 'BowlingTeam'] = team2
        match_data['BowlingTeam'] = team2
    else:
        match_data.at[index, 'BowlingTeam'] = team1
        match_data['BowlingTeam'] = team1

print(match_data.head(3))

data = match_data.groupby(['Match_Id' , 'Innings']).sum()['Total_Run'].reset_index() #reset_index , it allows you reset the index back to the default 0, 1, 2 etc indexes.
# print(data.head(10))

data = data[data['Innings'] == 1]
# print(data.head())

final_data = match.merge(data[['Match_Id' , 'Total_Run']] , left_on = 'Match_Id' , right_on = 'Match_Id')
# print(final_data['Team1'].unique())
# print(final_data.head(10))

teams = [
    'Sunrisers Hyderabad' ,
    'Mumbai Indians' ,
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders' ,
    'Delhi Capitals' ,
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals'
]

final_data['Team1'] == final_data['Team1'].str.replace('Delhi Daredevils' , 'Delhi Capitals')
final_data['Team2'] == final_data['Team2'].str.replace('Delhi Daredevils' , 'Delhi Capitals')

final_data['Team1'] == final_data['Team1'].str.replace('Deccan Chargers' ,  'Sunrisers Hyderabad')
final_data['Team2'] == final_data['Team2'].str.replace('Deccan Chargers' ,  'Sunrisers Hyderabad')

final_data = final_data[final_data['Team1'].isin(teams)]
final_data = final_data[final_data['Team2'].isin(teams)]

final_data = final_data[['Match_Id' , 'City' , 'WinningTeam' , 'Total_Run']]
# print(final_data.head())
delivery_data = final_data.merge(match_data , on = 'Match_Id')
delivery_data = delivery_data[delivery_data['Innings'] == 2]
# print(delivery_data.head())

grouped = delivery_data.groupby('Match_Id')
wickets = grouped.Total_Run_y.cumsum()
delivery_data['Current_Score'] = grouped['Total_Run_y'].cumsum()
# delivery_data['Wicket_Left'] = 10 - wickets


delivery_data['Runs_Left'] = delivery_data['Total_Run_x'] - delivery_data['Current_Score']
delivery_data['Balls_Left'] = 120 - (delivery_data['Over'] * 6 + delivery_data['Ball'])
delivery_data['Player_Dismissed'] = delivery_data['Player_Dismissed'].fillna('0')
delivery_data['Player_Dismissed'] = delivery_data['Player_Dismissed'].apply(lambda x:x if x == '0'  else '1')

delivery_data['Total_Run_x'] = p.to_numeric(delivery_data['Total_Run_x'], errors='coerce')
delivery_data['Total_Run_y'] = p.to_numeric(delivery_data['Total_Run_y'], errors='coerce')
delivery_data['Over'] = p.to_numeric(delivery_data['Over'], errors='coerce')
delivery_data['Ball'] = p.to_numeric(delivery_data['Ball'], errors='coerce')
delivery_data['Player_Dismissed'] = p.to_numeric(delivery_data['Player_Dismissed'] , errors = 'coerce')

grouped = delivery_data.groupby('Match_Id')
wickets = grouped.Player_Dismissed.cumsum()
delivery_data['Wicket_Left'] = grouped['Player_Dismissed'].cumsum()
delivery_data['Wicket_Left'] = 10 - wickets

delivery_data['Current_Run_Rate'] = delivery_data['Current_Score']/(120 - delivery_data['Balls_Left'])
delivery_data['Required_Run_Rate'] = (delivery_data['Runs_Left']*6) / delivery_data['Balls_Left']

def result(row):
    return 1 if row['BattingTeam'] == row['WinningTeam'] else 0

delivery_data['Result'] = delivery_data.apply(result , axis = 1)

final_df = delivery_data[['BattingTeam' , 'BowlingTeam' , 'City' , 'Runs_Left' , 'Balls_Left' , 'Wicket_Left' , 'Total_Run_x' , 'Current_Run_Rate' , 'Required_Run_Rate' , 'Result']]
print(final_df.head())
