import random
import logging


class Sudoku:
    def __init__(self, order: int, grid=None) -> None:
        logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)

        self.set_order(order)
        self.reset()
        if grid is not None:
            self.init_from_grid(grid)

    def __eq__(self, other: "Sudoku") -> bool:
        """
        Check whether rows, cols and blocks are equal to the input object
        """
        for (i, el) in enumerate(self.row_dicts):
            if el != other.row_dicts[i]:
                return False
        for (i, el) in enumerate(self.col_dicts):
            if el != other.col_dicts[i]:
                return False
        for (i, el) in enumerate(self.blk_dicts):
            if el != other.blk_dicts[i]:
                return False
        # Reached here, all good
        return True

    def __str__(self) -> str:
        retstr = ""
        for i in range(self.n):
            row_str = ""
            for j in range(self.n):
                if j not in self.row_dicts[i]:
                    row_str += "* "
                else:
                    row_str += str(self.row_dicts[i][j]) + " "
                if j % self.order == (self.order - 1) and j < (self.n - 1):
                    row_str += "| "
            retstr += row_str + "\n"
            if i % self.order == (self.order - 1) and i < (self.n - 1):
                retstr += "-" * (self.n * 2 + self.order) + "\n"
        return retstr

    def set_order(self, order: int) -> None:
        self.order = order
        self.n = order * order
        self.n_blocks = order * order

    def get_block_ind(self, row_ind: int, col_ind: int) -> tuple[int, int]:
        block_ind = (row_ind // self.order) * \
            self.order + col_ind // self.order
        block_el_ind = (row_ind % self.order) * \
            self.order + col_ind % self.order
        return (block_ind, block_el_ind)

    def allowed_vals(self, row_ind: int, col_ind: int) -> list[int]:
        (bl_ind, _) = self.get_block_ind(row_ind, col_ind)
        filled_vals_row = self.row_dicts[row_ind].values()
        filled_vals_col = self.col_dicts[col_ind].values()
        filled_vals_blk = self.blk_dicts[bl_ind].values()
        bt_key = (row_ind, col_ind)
        if bt_key in self.backtracked:
            backtracked_vals = self.backtracked[bt_key]
        else:
            backtracked_vals = []
        retvals = [i for i in range(1, self.n + 1) if i not in filled_vals_row and
                   i not in filled_vals_col and
                   i not in filled_vals_blk and
                   i not in backtracked_vals]
        return retvals

    def set_backtrack_val(self, row_ind: int, col_ind: int, val: int) -> None:
        bt_key = (row_ind, col_ind)
        if bt_key not in self.backtracked:
            self.backtracked[bt_key] = [val]
        else:
            self.backtracked[bt_key].append(val)

    def reset(self) -> None:
        self.row_dicts = [{} for _ in range(self.n)]
        self.col_dicts = [{} for _ in range(self.n)]
        self.blk_dicts = [{} for _ in range(self.n_blocks)]
        self.backtracked = {}

    def init_from_grid(self, grid: str | list[list[int]]):
        # in the grid parameter is a string of numbers, we convert into list of int
        if isinstance(grid, str):
            for i, c in enumerate(grid):
                row_ind = i // self.n
                col_ind = i % self.n
                val = int(c)
                if val > 0 and val <= self.n:
                    self.set_val(row_ind, col_ind, val)
        else:
            for (row_ind, row) in enumerate(grid):
                for (col_ind, val) in enumerate(row):
                    if val > 0 and val <= self.n:
                        self.set_val(row_ind, col_ind, val)

    def set_val(self, row_ind: int, col_ind: int, new_val: int) -> None:
        (bl_ind, bl_el_ind) = self.get_block_ind(row_ind, col_ind)
        self.blk_dicts[bl_ind][bl_el_ind] = new_val
        self.row_dicts[row_ind][col_ind] = new_val
        self.col_dicts[col_ind][row_ind] = new_val

    def del_val(self, row_ind: int, col_ind: int) -> None:
        if col_ind in self.row_dicts[row_ind]:
            (bl_ind, bl_el_ind) = self.get_block_ind(row_ind, col_ind)
            del self.blk_dicts[bl_ind][bl_el_ind]
            del self.row_dicts[row_ind][col_ind]
            del self.col_dicts[col_ind][row_ind]

    def get_vals(self) -> list[list[int]]:
        retlist = [
            [-1 for _ in range(self.n)]
            for _ in range(self.n)
        ]

        for row, row_dict in enumerate(self.row_dicts):
            for col, val in row_dict.items():
                retlist[row][col] = val

        return retlist

    def generate_elem(self, row_ind: int, col_ind: int) -> bool:
        allowed_vals = self.allowed_vals(row_ind, col_ind)
        if len(allowed_vals) > 0:
            # assign new value
            new_val = random.choice(allowed_vals)
            self.set_val(row_ind, col_ind, new_val)
            return True
        else:
            return False

    def generate_puzzle(self) -> None:
        """
        Generates the sudoku puzzle, with missing elements that can be filled
        in a unique manner to complete it-
        TODO: make use of the generation function below and take off elements
        according to a difficulty grade.
        """
        self.reset()
        if self.order == 3:
            puzzles = [
                "004300209005009001070060043006002087190007400050083000600000105003508690042910300",
                "040100050107003960520008000000000017000906800803050620090060543600080700250097100",
                "600120384008459072000006005000264030070080006940003000310000050089700000502000190",
                "497200000100400005000016098620300040300900000001072600002005870000600004530097061",
                "005910308009403060027500100030000201000820007006007004000080000640150700890000420",
                "100005007380900000600000480820001075040760020069002001005039004000020100000046352",
                "009065430007000800600108020003090002501403960804000100030509007056080000070240090",
                "000000657702400100350006000500020009210300500047109008008760090900502030030018206",
                "503070190000006750047190600400038000950200300000010072000804001300001860086720005",
                "060720908084003001700100065900008000071060000002010034000200706030049800215000090",
                "004083002051004300000096710120800006040000500830607900060309040007000205090050803",
                "000060280709001000860320074900040510007190340003006002002970000300800905500000021",
                "004300000890200670700900050500008140070032060600001308001750900005040012980006005",
                "008070100120090054000003020604010089530780010009062300080040607007506000400800002",
                "065370002000001370000640800097004028080090001100020940040006700070018050230900060",
                "005710329000362800004000000100000980083900250006003100300106000409800007070029500",
                "200005300000073850000108904070009001651000040040200080300050000580760100410030096",
                "040800500080760092001005470056309000009001004320500010000200700700090030005008026",
                "050083017000100400304005608000030009090824500006000070009000050007290086103607204",
                "700084005300701020080260401624109038803600010000000002900000000001005790035400006",
                "067050010084309000003080040090000205000621790700093600300400000020007153500800076",
                "001409030000306052007008190060020800000003065894507000403091080079040026000700900",
                "206030000001065070047108050500000029008019406000420001000042800609300005070000013",
                "004502178100090030000800004600450000070900012801203500400000009350060807090300620",
                "140060800085010040907400250030070400209000307008900060000740010601305090700002600",
                "590000147000900008072000030700040290020030806800170050005764009036005000100800002",
                "100000090208970605000532000006050400700806002083700010604080120890600050015040007",
                "900084060604005207030070080760001500053000001000409603105026090002040000800003710",
                "308056007006900253012040000000000320904800000760109805000001904831000506040007030",
                "170300009008040600000060030600800001924600300300902500010200040709503016005007800",
                "004030021070005009380690000030000000602100450010907003000846700560001240008250030",
                "083200096200030704007915000402390008010004060069870000000400007500060280070050900",
                "803000270409008000700024096000006915001802000030750000054000060608100003372009140",
                "070490103003070590050000000000000061100749020024306008600980700012600000480007052",
                "830040096020010008904700030409002065308001070000603800507030020000506400002080100",
                "060250000792006100000081600009000500410009780207300004000763010300540290800000040",
                "050400680090100000008059302007203000000600208604080005036004190100007000072800050",
                "010092047000700609600040100003000000720008900840105070106400280480030000900017005",
                "078010609203009008410060052720106030000400700091305000932000400005720010000008006",
                "056010000280030040040090765790003008005760000000004001100600203020001400060805900",
                "000001847010000000059348006300020004076100500200006790040207069007800415003590000",
                "205040003001009000046001587004607090802000056090020340170008200000500800500903001",
                "103800000906400072000205090090070050084901300002506000210004005308700401070080600",
                "850420370003000010000170009000500602029304000010000438046090805005000900702840003",
                "061490020280007050003108007600704031000250074090600000000010008570000206800906000",
                "608900050000320190010000300400073600570260000003105020080000064020090507047008001",
                "000700900004300527010006084800094053040001200962080070100869000700020130059000000",
                "680905000003000508402108703390720800000000010045006900060804002001002075700013000",
                "600837001089004700102000400000450020030609005040000860908006070700098010005100930",
                "000067430800009150500003009007000010001806304940350020009010502608200700400708000",
                "063700401400000000700091300092076030004500260035000100509040820087010609000003000",
                "000340002006082073700100450082005014000098300670000005140700000905030020030000806",
                "008070600960001405402000010200830090600790103007004026500900307030020500000310089",
                "087200490060891000005000002400300500019002803000706200030050600500060017071403050",
                "107008000650100000300060072060030250480009700001407009000000800003980015040203060",
                "000704010803620907016090000000100406598400002000030079034075000100900360250300040",
                "500038701026004850300072940054100300890400000000060000003000069000721400401390000",
                "600050000073008020854027000201700530400069007080000900027301084060540009300000001",
                "000900406701040002009501378012300009300004080085000230050007000093008600020600541",
                "007500904000082305001600002800036070016004200430190050540008000029071030000000609",
                "001040730630900004075000200000000501069001000004002079980500602710609300000203080",
                "507108002000043100900500006070050004100002069600700380320490000001006400009070528",
                "807000000610005430400690000002800709003007820900051046000009670054000000200403018",
                "040000200710930000058020760300060000406800920020705800031070090007058601500100007",
                "002834700010060005798200400903050100000009007080072060600081024051000300040000090",
                "431800006000300010000006205609134070020000040000570089003659020500080104807000003",
                "020000106610239040050004007003520009009400080800701400795006002200180000400000653",
                "003000150620700009051000007180670003509003400030080201000207904000098620004510008",
                "400053270020600803008904010145200006000048300000001090601300450000070900780000060",
                "004030009520001300019860200402000700037006800608510020900080070073002046000000130",
                "023000061000068007000304058058006120007090080140000509000430000064085930001072000",
                "031020060005074003006800020700406038000095706200780901098140000500000890300000002",
                "015040002020560098300010007200000600940001000030680704458000000090872050600430900",
                "604302001780006300900510400003004078028000096000601004090057800000020015000403700",
                "000048010943050700008200006050000379002400080070013040600590002109087500530060000",
                "900003050246050700070680100000008039501200800700006001809420006002010078030000240",
                "002078000006015908905000100400709500030054070000061204000020306008000009710800450",
                "043020000050180070890506100400009706031000059000871000070005460002043501000600320",
                "206597403080103000507000009000004210028006500409010060700305000001200000300480902",
                "270600050000070406006059030040005600081000040029006173390000002000097800807140005",
                "010907030002100004958002000600030002073060580000705069080000406304000105009210800",
                "290041000470302050000060208039400005100000070504100603613200704000003080005900100",
                "080623090400007000900000713006910400805000026014500007030708000500000009021460085",
                "000050007050004020020160430941000000008746000000300508400000209705080300219030076",
                "000457602940061300070000080100509400700086930020003007400010059006000800218300700",
                "000720041003051890070000000730200160004109305205080009800000006090007200560904003",
                "000230050673400800005007900310780000064009200082546009008000103000801700000600405",
                "980046025000090700700300004008023010045070008000105006014800900506000307090002600",
                "200000001003060008807031940002506070409800056100000380038670500705090263000004000",
                "004206050070000306000075120002608710890000040047090000000000590056002000009307208",
                "090604025000500904503700100002059001010000000006008437080003600400020000700000058",
                "000750600061980004400000720259006000803000010000820075090208000010060490007340058",
                "600000045200009803089007000001402309790300050000050080076030400520000700004086012",
                "020980040030047601019006080700490000800023907000605000904800006001000300350014020",
                "008060700030870012000205930000700504905004000802903070106000080009120360700000250",
                "600300000502809070013050002700240085004007126009508000305000400170400003008090607",
                "610030004005008703040906020009200400000403680002015070700001500003680200090070031",
                "002080500058370100700006039005902700000000264030410000087201605901040080000090020",
            ]
            ind = random.randint(0, len(puzzles) - 1)
            self.init_from_grid(puzzles[ind])

    def generate(self) -> None:
        """
        Generates a complete sudoku, with no missing elements
        """
        self.reset()
        generate_success = False
        generate_cnt = 0
        while not generate_success:
            generate_success = True
            for i in range(self.n * self.n):
                row_ind = i // self.n
                col_ind = i % self.n
                # if empty, we add
                if col_ind not in self.row_dicts[row_ind]:
                    if not self.generate_elem(row_ind, col_ind):
                        # backtrack
                        bt_success = False
                        for b_i in reversed(range(i)):
                            b_row_ind = b_i // self.n
                            b_col_ind = b_i % self.n
                            b_val = self.row_dicts[b_row_ind][b_col_ind]
                            self.set_backtrack_val(b_row_ind, b_col_ind, b_val)
                            b_allowed_vals = self.allowed_vals(
                                b_row_ind, b_col_ind)
                            while len(b_allowed_vals) > 0:
                                b_val = random.choice(b_allowed_vals)
                                self.set_val(b_row_ind, b_col_ind, b_val)
                                if self.generate_elem(row_ind, col_ind):
                                    bt_success = True
                                    break
                                else:
                                    self.set_backtrack_val(
                                        b_row_ind, b_col_ind, b_val)
                                    b_allowed_vals = self.allowed_vals(
                                        b_row_ind, b_col_ind)
                            if bt_success:
                                break
                        if bt_success:
                            # clear backtracked values for the next iteration, if need be
                            self.backtracked = {}
                        else:
                            logging.warning("Backtrack failed, trying again")
                            self.reset()
                            generate_success = False
                            generate_cnt += 1
                            break
        logging.info(
            "Puzzle generated after {} iterations".format(generate_cnt))

    def sanity_checks(self, rcb: list[dict[int, int]]) -> bool:
        # check if rows, cols and blocks are unique
        for el in rcb:
            vals = el.values()
            if len(set(vals)) != len(vals) or \
               min(vals) < 0 and max(vals) > self.n:
                return False
        return True

    def verify(self) -> bool:
        return self.sanity_checks(self.row_dicts) and \
            self.sanity_checks(self.col_dicts) and \
            self.sanity_checks(self.blk_dicts)

    def solve(self) -> bool:
        # construct matrix of unsolved row, col tuples
        unsolved_row_col = []
        for row, row_dict in enumerate(self.row_dicts):
            for col in range(self.n):
                if col not in row_dict:
                    unsolved_row_col += [(row, col)]

        unsolved_row_col_next = list(unsolved_row_col)
        at_least_one = True
        while len(unsolved_row_col_next) > 0:
            at_least_one = False
            for row, col in unsolved_row_col:
                allowed_vals = self.allowed_vals(row, col)
                if len(allowed_vals) == 1:
                    self.set_val(row, col, allowed_vals[0])
                    unsolved_row_col.remove((row, col))
                    at_least_one = True
            unsolved_row_col_next = list(unsolved_row_col)
            if not at_least_one:
                break

        return len(unsolved_row_col_next) == 0


if __name__ == "__main__":
    sdk = Sudoku(3)
    sdk.generate()
    print(sdk)
