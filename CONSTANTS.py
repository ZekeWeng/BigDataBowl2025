### PATHES
INPUT_PATH = "data/2025/raw"
OUTPUT_PATH = "data/2025/processed"

GAMES = "data/2025/raw/games.csv"
PLAYERS = "data/2025/raw/players.csv"
PLAYS = "data/2025/raw/plays.csv"
PLAYER_PLAY = "data/2025/raw/player_play.csv"
PLAYER_PLAY = "data/2025/raw/player_play.csv"



### PRESNAP
PRESNAP_PLAYS_COLS = [
    'gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'absoluteYardlineNumber',
    'playDescription',
    'possessionTeam', 'defensiveTeam', 'gameClock', 'playClockAtSnap', 'expectedPoints',
    'preSnapHomeScore', 'preSnapVisitorScore', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability',
    'offenseFormation', 'receiverAlignment', 'penaltyYards', 'qbSpike', 'qbKneel',
    'isDropback', 'pff_passCoverage', 'pff_manZone'
]
PRESNAP_PLAYERS_COLS= [
    'nflId', 'displayName', 'position', 'height', 'weight'
]

PRESNAP_PLAYER_PLAY_COLS = [
    'gameId', 'playId', 'nflId', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset',
]