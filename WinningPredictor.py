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
print(match_data.head())