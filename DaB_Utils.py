def ANSI_string(s="", color=None, background=None, bold=False):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    background_colors = {
        'black': '\033[40m',
        'red': '\033[41m',
        'green': '\033[42m',
        'yellow': '\033[43m',
        'blue': '\033[44m',
        'magenta': '\033[45m',
        'cyan': '\033[46m',
        'white': '\033[47m',
        'reset': '\033[0m',
        'gray': '\033[100m',  # 新增的灰色背景
        'light_gray': '\033[47m'  # 新增的淺灰色背景
    }

    styles = {
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    color_code = colors[color] if color in colors else ''
    background_code = background_colors[background] if background in colors else ''
    bold_code = styles['bold'] if bold else ''

    return f"{color_code}{background_code}{bold_code}{s}{styles['reset']}"


def isValid(board, r, c):
    if board[r][c] == 0:
        return True
    else:
        return False

# 取得合法部
def getValidMoves(board):
    ValidMoves = []
    m = len(board)
    n = len(board[0])
    for i in range(m):
        for j in range(n):
            if board[i][j] == 0:
                ValidMoves.append((i, j))
    return ValidMoves

# 確認有無得分，若有得分也將區塊置換成player區塊
def checkBox(board, player):
    box_filled = False
    score = 0
    m = (len(board)+1) // 2
    n = (len(board[0])+1) // 2
    for i in range(m-1):
        for j in range(n-1):
            box_i = 2*i + 1
            box_j = 2*j + 1
            # 檢查該方格的四條邊是否都不為 0
            if (board[box_i][box_j] == 8 and
                board[box_i-1][box_j] != 0 and
                board[box_i+1][box_j] != 0 and
                board[box_i][box_j-1] != 0 and
                    board[box_i][box_j+1] != 0):

                # 更新該方格的狀態
                board[box_i][box_j] += player
                score += 1
                box_filled = True

    # 若player有得分，則回傳True
    return box_filled, score

def make_move(board, r, c, player):
    score = 0
    if isValid(board,r,c):
        board[r][c] = player
    else:
        print("Invalid move!")
        return player, score
    # 若沒得分就換人
    box_filled, score = checkBox(board, player)
    if not box_filled:
        player *= -1

    # 回傳下完後的玩家以及是否得一分
    return player, score

def isGameOver(board):
    if len(getValidMoves(board)) == 0:
        return True
    return False

def GetWinner(board, p1_p2_scores):
    if isGameOver(board):
        if p1_p2_scores[0] > p1_p2_scores[1]:
            return -1
        elif p1_p2_scores[0] < p1_p2_scores[1]:
            return 1
        else:
            return 0
    return None


def print_board(board, board_rows, board_cols):
    for i in range(board_rows):
        for j in range(board_cols):
            value = board[i][j]
            if value == 0:
                print(' ', end='')
            elif value in [-1, 1]:  # 處理先手和後手邊
                color = 'blue' if value == -1 else 'red'
                if i % 2 == 0:  # 偶數列 -> 水平線
                    print(ANSI_string(s='-', color=color), end='')
                else:  # 奇數列 -> 垂直線
                    print(ANSI_string(s='|', color=color), end='')
            elif value == 5:  # 頂點
                print('o', end='')
            elif value in [7, 9]:  # 處理先手和後手佔領
                background = 'blue' if value == 7 else 'red'
                print(ANSI_string(s=str(value), background=background), end='')
            elif value == 8:
                print(' ', end='')
        print()
    print("="*(board_cols))