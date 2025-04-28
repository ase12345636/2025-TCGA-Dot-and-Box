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