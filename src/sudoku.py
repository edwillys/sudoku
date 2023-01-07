import random
import logging
import math


class Sudoku:
    def __init__(self, order: int = 3, grid=None) -> None:
        logging.basicConfig(
            format='%(levelname)s:%(message)s', level=logging.DEBUG)
        if grid is not None:
            grid_order = math.sqrt(math.sqrt(len(grid)))
            if round(grid_order) == grid_order:
                self.set_order(int(grid_order))
                self.reset()
                self.init_from_grid(grid)
            else:
                self.set_order(order)
                self.reset()
                logging.error(
                    "Grid is not square, falling back to {}".format(order))
        else:
            self.set_order(order)
            self.reset()

    def __eq__(self, other: object) -> bool:
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

    def get_block_ind(self, row_ind: int, col_ind: int) -> tuple(int, int):
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

    def init_from_grid(self, grid: str):
        # in the grid parameter is a string of numbers, we convert into list of int
        if isinstance(grid, str):
            grid = [int(num) for num in grid]
        for (i, val) in enumerate(grid):
            if val > 0 and val <= self.n:
                row_ind = i // self.n
                col_ind = i % self.n
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

    def generate_elem(self, row_ind: int, col_ind: int) -> bool:
        allowed_vals = self.allowed_vals(row_ind, col_ind)
        if len(allowed_vals) > 0:
            # assign new value
            new_val = random.choice(allowed_vals)
            self.set_val(row_ind, col_ind, new_val)
            return True
        else:
            return False

    def generate(self) -> None:
        # generate diagonal blocks: 0, n + 1, 2 * (n + 1), ...
        # diag_bl_inds = np.linspace(0, self.n_blocks - 1, self.order, dtype=int)
        # for bl_ind in diag_bl_inds:
        #    perm = np.random.permutation(range(1, self.n + 1))
        #    for (k, p) in enumerate(perm):
        #        off = (bl_ind % self.order) * self.order
        #        self.row_dicts[off + k // self.order][off + k % self.order] = p
        #        self.col_dicts[off + k % self.order][off + k // self.order] = p
        #        self.blk_dicts[bl_ind][k] = p

        # generate missing entries, block based
        # missing_bl_inds = [i for i in range(self.n_blocks - 1) if (i % (self.order + 1)) != 0]
        # for bl_ind in missing_bl_inds:
        #    # A block has N entries
        #    for k in range(self.n):
        #        row_off = (bl_ind // self.order) * self.order
        #        col_off = (bl_ind % self.order) * self.order
        #        row_ind = row_off + k // self.order
        #        col_ind = col_off + k % self.order
        #        allowed_vals = self.allowed_vals(row_ind, col_ind)
        #        if len(allowed_vals):
        #            # assign new value
        #            new_val = np.random.choice(allowed_vals)
        #            self.blk_dicts[bl_ind][k] = new_val
        #            self.row_dicts[row_ind][col_ind] = new_val
        #            self.col_dicts[col_ind][row_ind] = new_val
        #        else:
        #            return
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

    def sanity_checks(self, rcb: dict[int]) -> bool:
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

    def solve(self):
        pass


if __name__ == "__main__":
    sdk = Sudoku()
    sdk.generate()
    print(sdk)
