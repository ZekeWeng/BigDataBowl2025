### PATHES
INPUT_PATH = "data/2025/raw"
OUTPUT_PATH = "data/2025/processed"


### PRESNAP
PRESNAP_PLAYS_COLS = [
    'gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'absoluteYardlineNumber',
    'playDescription',
    'possessionTeam', 'defensiveTeam', 'gameClock', 'playClockAtSnap', 'expectedPoints',
    'preSnapHomeScore', 'preSnapVisitorScore', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability',
    'offenseFormation', 'receiverAlignment',
    'isDropback', 'pff_passCoverage', 'pff_manZone'
]
PRESNAP_PLAYERS_COLS= [
    'nflId', 'displayName', 'position', 'height', 'weight'
]
# Using all Tracking Cols
PRESNAP_PLAYER_PLAY_COLS = [
    'gameId', 'playId', 'nflId', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset',
]

### MATCHUP
MATCHUP_PLAYS_COLS = [
    'gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'absoluteYardlineNumber', 'homeTeamAbbr', 'visitorTeamAbbr', 'club', 'playDirection'
    'playDescription',
    'possessionTeam', 'defensiveTeam', 'gameClock', 'playClockAtSnap', 'expectedPoints',
    'preSnapHomeScore', 'preSnapVisitorScore', 'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability',
    'offenseFormation', 'receiverAlignment',
    'pff_runConceptPrimary', 'pff_runConceptSecondary', 'pff_runPassOption', 'pff_passCoverage', 'pff_manZone',
    'yardsGained'
]
MATCHUP_PLAYERS_COLS= [
    'nflId', 'displayName', 'position', 'height', 'weight'
]
# Using all Tracking Cols
MATCHUP_PLAYER_PLAY_COLS = [
    'gameId', 'playId', 'nflId', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset',
    'pff_defensiveCoverageAssignment', 'pff_primaryDefensiveCoverageMatchupNflId', 'pff_secondaryDefensiveCoverageMatchupNflId'
]