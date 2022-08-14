use std::convert::{TryFrom, TryInto};
use std::hash::Hash;
use std::{collections::HashSet, usize};

use itertools::Itertools;
#[allow(unused_imports)]
#[cfg(feature = "local")]
use log::{debug, error, info, warn, LevelFilter};

use petgraph::Direction;
use rand::{prelude::SliceRandom, Rng};

#[macro_export]
macro_rules! input_value {
    ($f: expr, $x: expr) => {
        let mut line = String::new();
        $f.read_line(&mut line).unwrap();
        $x = line.trim().parse().unwrap();
    };
}

#[macro_export]
macro_rules! input_vec {
    ($f: expr, $x: expr) => {
        let mut line = String::new();
        $f.read_line(&mut line).unwrap();
        let mut iter = line.split_whitespace();
        $x = iter.map(|v| v.parse().unwrap()).collect();
    };
}

struct Logger {}

#[cfg(feature = "local")]
impl Logger {
    // loggingの記述
}

#[cfg(not(feature = "local"))]
impl Logger {
    // 何もしないメソッドの実装
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        ms - STIME
    }
}

fn manhattan_distance(x: [usize; 2], y: [usize; 2]) -> usize {
    abs_diff(x[0], y[0]) + abs_diff(x[1], y[1])
}

fn abs_diff(x: usize, y: usize) -> usize {
    if x >= y {
        return x - y;
    } else {
        return y - x;
    }
}

struct Unionfind {
    parents: Vec<i64>,
    size: usize,
}

impl Unionfind {
    pub fn new(n: usize) -> Self {
        Unionfind {
            parents: vec![-1; n * n], // 負値:: -size  正値:: parent を示す
            size: n,
        }
    }

    fn find(&mut self, x: (usize, usize)) -> usize {
        let x = x.0 * self.size + x.1;
        self._find(x)
    }

    fn _find(&self, x: usize) -> usize {
        if self.parents[x] < 0 {
            return x;
        } else {
            return self._find(self.parents[x] as usize);
        }
    }

    fn unite(&mut self, x: (usize, usize), y: (usize, usize)) {
        let mut x_root: usize = self.find(x);
        let mut y_root: usize = self.find(y);

        if x_root == y_root {
            return;
        }
        if self.parents[x_root] > self.parents[y_root] {
            std::mem::swap(&mut x_root, &mut y_root)
        }
        self.parents[x_root] += self.parents[y_root];
        self.parents[y_root] = x_root as i64;
    }
}

struct Answer {
    moves: Vec<[usize; 4]>,
    connects: Vec<[usize; 4]>,
}

impl Answer {
    pub fn new(moves: &Vec<[usize; 4]>, connects: &Vec<[usize; 4]>) -> Self {
        Answer {
            moves: moves.clone(),
            connects: connects.clone(),
        }
    }

    fn arr2string(arr: &[usize; 4]) -> String {
        arr.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(" ")
    }

    fn print_answer(&self) {
        println!("{}", self.moves.len());
        for arr in self.moves.iter() {
            let ans = Answer::arr2string(arr);
            print!("{}\n", ans);
        }

        println!("{}", self.connects.len());
        for arr in self.connects.iter() {
            println!("{}", Answer::arr2string(arr))
        }
    }
}

struct Board {
    c: Vec<Vec<usize>>,
    k: usize,
    n: usize,
}

impl Board {
    pub fn new(c: &Vec<Vec<usize>>, k: usize) -> Self {
        Board {
            c: c.clone(),
            k,
            n: c.len(),
        }
    }

    pub fn show_board(&self) {
        for c in self.c.iter() {
            let output = c
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(" ");

            println!("{}", output);
        }
    }

    fn count_number_in_area(&self, center: [usize; 2], distance: usize, target: usize) -> usize {
        // center から distandce 内にある target の個数を返す。 マンハッタン距離で計算
        let start_h = center[0].checked_sub(distance).unwrap_or(0);
        let start_w = center[1].checked_sub(distance).unwrap_or(0);

        let n = self.c.len();
        let end_h = std::cmp::min(center[0] + distance + 1, n);
        let end_w = std::cmp::min(center[1] + distance + 1, n);
        let mut cnt: usize = 0;

        for h in start_h..end_h {
            for w in start_w..end_w {
                if abs_diff(h, center[0]) + abs_diff(w, center[1]) > distance {
                    continue;
                }
                if self.c[h][w] == target {
                    cnt += 1
                }
            }
        }
        cnt
    }

    pub fn score_count_num_in_area(&self, computers_type: [usize; 5]) -> usize {
        let point = [
            [0, 0],
            [0, self.n - 1],
            [self.n - 1, 0],
            [self.n - 1, self.n - 1],
            [self.n / 2, self.n / 2],
        ];

        let mut score = 0;
        for (computer_type, center) in computers_type.iter().zip(point.iter()) {
            score += self.count_number_in_area(center.clone(), self.n / 2, computer_type.clone());
        }

        score
    }

    pub fn make_init_point(&self) -> [usize; 5] {
        fn dfs(arr: Vec<usize>, k: usize, conditions: &mut Vec<[usize; 5]>) {
            if arr.len() == 5 {
                let mut is_in: Vec<bool> = vec![false; k];

                for a in arr.iter() {
                    is_in[a - 1] = true;
                }

                if is_in.iter().all(|&x| x == true) {
                    conditions.push([arr[0], arr[1], arr[2], arr[3], arr[4]]);
                }
                return;
            }

            for computer_type in 1..=k {
                let mut _arr = arr.clone();
                _arr.push(computer_type);
                dfs(_arr, k, conditions);
            }
        }

        let mut conditions: Vec<[usize; 5]> = Vec::new();

        let arr: Vec<usize> = Vec::new();
        dfs(arr, self.k, &mut conditions);

        let mut max_score = 0;
        let mut ans_condition: [usize; 5] = [1; 5];
        let point = [
            [0, 0],
            [0, self.n - 1],
            [self.n - 1, 0],
            [self.n - 1, self.n - 1],
            [self.n / 2, self.n / 2],
        ];

        for &v in conditions.iter() {
            let score = self.score_count_num_in_area(v);

            if score > max_score {
                max_score = score;
                ans_condition = v.clone();
            }
        }

        ans_condition
    }

    pub fn make_init_bord(&self) -> Vec<Vec<usize>> {
        let ans_condition = self.make_init_point();

        let n = self.c.len();

        let mut res_board: Vec<Vec<usize>> = vec![vec![1; n]; n];

        let point = [
            [0, 0],
            [0, n - 1],
            [n - 1, 0],
            [n - 1, n - 1],
            [n / 2, n / 2],
        ];

        for (p, &computer_type) in point.iter().zip(ans_condition.iter()) {
            let start_h = p[0].checked_sub(n / 2).unwrap_or(0);
            let start_w = p[1].checked_sub(n / 2).unwrap_or(0);

            let end_h = std::cmp::min(p[0] + n / 2, n);
            let end_w = std::cmp::min(p[1] + n / 2, n);

            for h in start_h..end_h {
                for w in start_w..end_w {
                    res_board[h][w] = computer_type;
                }
            }
        }
        res_board
    }
}

struct Status {
    moves: Vec<[usize; 4]>,
    connects: Vec<[usize; 4]>,
    c: Vec<Vec<usize>>,
    n: usize,
    k: usize,
    move_count: usize,
}

impl Status {
    pub fn new(c: &Vec<Vec<usize>>, n: usize, k: usize) -> Self {
        Status {
            moves: Vec::new(),
            connects: Vec::new(),
            c: c.clone(),
            n,
            k,
            move_count: 0,
        }
    }

    pub fn solve(&mut self, v: [usize; 5]) {
        // self._move();
        self._move_close_to_points(v);
        self.connect();
    }
    fn _move(&mut self) {
        let move_direction: [[i64; 2]; 4] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        let move_limit = self.k * 100 / 2;
        let mut rng = rand::thread_rng();

        for _ in 0..move_limit {
            let i: usize = rng.gen_range(0, self.n);
            let j: usize = rng.gen_range(0, self.n);

            if self.c[i][j] == 0 {
                continue;
            }
            // let v = rng.choose(&move_direction).unwrap();
            let v = move_direction.choose(&mut rng).unwrap();
            let i2 = i as i64 + v[0];
            let j2 = j as i64 + v[1];
            if i2 < 0 || i2 >= self.n as i64 || j2 < 0 || j2 >= self.n as i64 {
                continue;
            }
            let i2 = i2 as usize;
            let j2 = j2 as usize;

            if self.c[i2][j2] != 0 {
                continue;
            }

            self.c[i2][j2] = self.c[i][j];
            self.c[i][j] = 0;
            self.moves.push([i, j, i2, j2]);
        }
    }

    fn _move_close_to_points(&mut self, v: [usize; 5]) {
        let move_direction: [[i64; 2]; 4] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        let move_limit = self.k * 100 / 2;
        let mut rng = rand::thread_rng();
        let time_limit: f64 = 2.8;

        let points = [
            [0, 0],
            [0, self.n - 1],
            [self.n - 1, 0],
            [self.n - 1, self.n - 1],
            [self.n / 2, self.n / 2],
        ];

        let mut move_cnt = 0;

        loop {
            if get_time() / time_limit > 1.0 {
                break;
            }

            let uh = rng.gen_range(0, self.n);
            let uw = rng.gen_range(0, self.n);

            if self.c[uh][uw] == 0 {
                continue;
            }

            let direction = move_direction.choose(&mut rng).unwrap();

            let vh = match usize::try_from(uh as i64 + direction[0]) {
                Ok(v) => v,
                _ => continue,
            };
            let vw = match usize::try_from(uw as i64 + direction[1]) {
                Ok(v) => v,
                _ => continue,
            };

            if vh >= self.n || vw >= self.n {
                continue;
            }

            if self.c[vh][vw] != 0 {
                continue;
            }

            let mut u_score: usize = 10000000;
            let mut v_score: usize = 10000000;

            for (i, &point) in points.iter().enumerate() {
                if v[i] != self.c[uh][uw] {
                    continue;
                }

                u_score = std::cmp::min(u_score, manhattan_distance([uh, uw], point));
                v_score = std::cmp::min(v_score, manhattan_distance([vh, vw], point));
            }

            if u_score > v_score {
                continue;
            }

            // move
            self.c[vh][vw] = self.c[uh][uw];
            self.c[uh][uw] = 0;

            self.moves.push([uh, uw, vh, vw]);

            move_cnt += 1;

            if move_cnt > move_limit {
                break;
            }
        }
    }

    fn connect(&mut self) {
        let used_cmputer: usize = 9;
        let connect_limit: usize = self.k * 100 - self.moves.len();

        for i in 0..self.n as usize {
            for j in 0..self.n as usize {
                if self.c[i][j] == 0 {
                    continue;
                };
                if self.c[i][j] == used_cmputer {
                    continue;
                }
                for &direction in [[0, 1], [1, 0]].iter() {
                    if self.can_connect([i, j], direction) {
                        self.do_connect([i, j], direction);
                    }
                    if self.connects.len() >= connect_limit {
                        return;
                    }
                }
            }
        }
        return;
    }

    fn can_connect(&self, pos: [usize; 2], direction: [usize; 2]) -> bool {
        let mut i = pos[0] + direction[0];
        let mut j = pos[1] + direction[1];
        while i < self.n && j < self.n {
            if self.c[i][j] == 0 {
                i += direction[0];
                j += direction[1];
            } else if self.c[i][j] == self.c[pos[0]][pos[1]] {
                return true;
            } else {
                return false;
            }
        }
        return false;
    }

    fn do_connect(&mut self, pos: [usize; 2], direction: [usize; 2]) {
        let mut i = pos[0] + direction[0];
        let mut j = pos[1] + direction[1];

        while i < self.n && j < self.n {
            if self.c[i][j] == 0 {
                self.c[i][j] = 9;
                continue;
            }
            if self.c[i][j] == self.c[pos[0]][pos[1]] {
                self.connects.push([pos[0], pos[1], i, j]);
                return;
            }
            i += direction[0];
            j += direction[1];
        }
    }
}

// 焼きなまし用

struct Anneal {
    c: Vec<Vec<usize>>,
    n: usize,
    k: usize,
    move_count: usize,
    cables: Vec<[usize; 4]>,
    connected_cables: HashSet<[usize; 4]>,
    cable_laying_position: Vec<Vec<usize>>,
}

impl Anneal {
    fn new(c: &Vec<Vec<usize>>, n: usize, k: usize) -> Self {
        let cable_laying_position: Vec<Vec<usize>> = vec![vec![0; n]; n];

        let mut anneal = Anneal {
            c: c.clone(),
            n: n,
            k: k,
            move_count: 0,
            cables: Vec::new(),
            connected_cables: HashSet::new(),
            cable_laying_position,
        };

        anneal.cables = anneal.get_all_cables();
        anneal
    }

    fn get_all_cables(&self) -> Vec<[usize; 4]> {
        let mut cables: Vec<[usize; 4]> = Vec::new();

        for h in 0..self.n {
            for w in 0..self.n {
                if self.c[h][w] == 0 {
                    continue;
                }

                for direction in [[0, 1], [1, 0]].iter() {
                    let mut vh = h;
                    let mut vw = w;

                    let dh = direction[0];
                    let dw = direction[1];
                    while vh < self.n - 1 && vw < self.n - 1 {
                        vh += dh;
                        vw += dw;
                        if self.c[h][w] == self.c[vh][vw] {
                            cables.push([h, w, vh, vw]);
                            break;
                        }
                        if self.c[vh][vw] != 0 {
                            break;
                        }
                    }
                }
            }
        }
        cables
    }

    fn calc_score(&self) -> usize {
        let mut uf = Unionfind::new(self.n);
        for cable in self.connected_cables.iter() {
            uf.unite((cable[0], cable[1]), (cable[2], cable[3]));
        }

        let mut computers: Vec<(usize, usize)> = Vec::new();
        for h in 0..self.n {
            for w in 0..self.n {
                if self.c[h][w] == 0 {
                    continue;
                }
                computers.push((h, w));
            }
        }

        let mut score = 0;
        for i in 0..computers.len() {
            for j in i + 1..computers.len() {
                if uf.find(computers[i]) != uf.find(computers[j]) {
                    continue;
                }
                if self.c[computers[i].0][computers[i].1] == self.c[computers[j].0][computers[j].1]
                {
                    score += 1;
                } else {
                    score -= 1;
                }
            }
        }

        return score;
    }

    fn is_cable_vertical(cable: [usize; 4]) -> bool {
        cable[1] == cable[3]
    }

    fn is_cable_crosses(u: [usize; 4], v: [usize; 4]) -> bool {
        // 両方縦方向の場合
        if Self::is_cable_vertical(u) && Self::is_cable_vertical(v) {
            // 存在する列が異なる
            if u[1] != v[1] {
                return false;
            }
            let upper = if u[0] < v[0] { u } else { v };
            let lower = if u[0] < v[0] { v } else { u };

            if upper[2] <= lower[0] {
                return false;
            } else {
                return true;
            }
        }

        // 両方横方向の場合
        if !Self::is_cable_vertical(u) && !Self::is_cable_vertical(v) {
            // 存在する列が異なる
            if u[0] != v[0] {
                return false;
            }
            let left = if u[1] < v[1] { u } else { v };
            let right = if u[1] < v[1] { v } else { u };

            if left[3] <= right[1] {
                return false;
            } else {
                return true;
            }
        }

        // 縦 横 の場合
        let vertical = if Self::is_cable_vertical(u) { u } else { v };
        let horizontal = if Self::is_cable_vertical(u) { v } else { u };

        // 開始点の重複がある場合はケーブルは交差していない
        let mut pc_point: HashSet<[usize; 2]> = HashSet::new();
        pc_point.insert([u[0], u[1]]);
        pc_point.insert([u[2], u[3]]);
        pc_point.insert([v[0], v[1]]);
        pc_point.insert([v[2], v[3]]);

        if pc_point.len() < 4 {
            return false;
        }

        if vertical[1] <= horizontal[3]
            && vertical[1] >= horizontal[1]
            && horizontal[0] <= vertical[3]
            && horizontal[0] >= vertical[0]
        {
            return true;
        }

        return false;
    }

    fn connect_cable(&mut self, cable: [usize; 4]) {
        if Self::is_cable_vertical(cable) {
            let w = cable[1];
            for h in std::cmp::min(cable[0], cable[2])..=std::cmp::max(cable[0], cable[2]) {
                self.cable_laying_position[h][w] += 1;
            }
        } else {
            let h = cable[0];
            for w in std::cmp::min(cable[1], cable[3])..=std::cmp::max(cable[1], cable[3]) {
                self.cable_laying_position[h][w] += 1;
            }
        }
        self.connected_cables.insert(cable.clone());
    }

    fn cut_cable(&mut self, cable: [usize; 4]) {
        if !self.connected_cables.contains(&cable) {
            return;
        }

        if Self::is_cable_vertical(cable) {
            let w = cable[1];
            let mut h1 = cable[0];
            let mut h2 = cable[2];
            if h1 > h2 {
                std::mem::swap(&mut h1, &mut h2);
            }

            for h in h1..=h2 {
                self.cable_laying_position[h][w] -= 1;
            }
        } else {
            let h = cable[0];
            let mut w1 = cable[1];
            let mut w2 = cable[3];
            if w1 > w2 {
                std::mem::swap(&mut w1, &mut w2);
            }
            for w in w1..=w2 {
                self.cable_laying_position[h][w] -= 1;
            }
        }
        self.connected_cables.remove(&cable);
    }

    fn can_connect_cable(&self, cable: [usize; 4]) -> bool {
        if Self::is_cable_vertical(cable) {
            let w = cable[1];
            for h in std::cmp::min(cable[0], cable[2]) + 1..std::cmp::max(cable[0], cable[2]) {
                if self.cable_laying_position[h][w] > 0 {
                    return false;
                }
            }
            return true;
        } else {
            let h = cable[0];
            for w in std::cmp::min(cable[1], cable[3]) + 1..std::cmp::max(cable[1], cable[3]) {
                if self.cable_laying_position[h][w] > 0 {
                    return false;
                }
            }
            return true;
        }
    }

    fn connect_cables_randomly(&mut self, connection_num: usize) -> Vec<[usize; 4]> {
        let mut rng = rand::thread_rng();

        let mut cables: Vec<[usize; 4]> = self.cables.clone();

        cables.shuffle(&mut rng);

        let mut connected_cable: Vec<[usize; 4]> = Vec::new();
        for cable in cables {
            if connected_cable.len() >= connection_num {
                break;
            }

            if self.connected_cables.contains(&cable) {
                continue;
            }
            if self.can_connect_cable(cable) {
                self.connect_cable(cable);
            }
            connected_cable.push(cable.clone());
        }
        connected_cable
    }

    fn cut_cables_randomly(&mut self, cut_num: usize) -> Vec<[usize; 4]> {
        let mut rng = rand::thread_rng();

        // todo vec を hash から作って回したほうが早い と思う

        let mut cables: Vec<[usize; 4]> = self.cables.clone();

        cables.shuffle(&mut rng);

        let mut cut_cnt: usize = 0;
        let mut cut_cable: Vec<[usize; 4]> = Vec::new();
        for cable in cables {
            if !self.connected_cables.contains(&cable) {
                continue;
            }

            self.cut_cable(cable);
            cut_cnt += 1;

            cut_cable.push(cable.clone());

            if cut_cnt >= cut_num {
                break;
            }
        }
        cut_cable
    }

    fn move_pc(&mut self, operation: [usize; 4]) {
        let [uh, uw, vh, vw] = operation;

        if self.c[uh][uw] == 0 {
            return;
        }

        if self.c[vh][vw] != 0 {
            return;
        }

        self.c[vh][vw] = self.c[uh][uw];
        self.c[uh][uw] = 0;
    }

    fn get_movable_position(&self, pc: [usize; 2]) -> Vec<[usize; 2]> {
        let directions = vec![(0, 1), (-1, 0), (0, -1), (1, 0)];

        let mut res = Vec::new();
        let [uh, uw] = pc;
        for (dh, dw) in directions {
            let vh = uh as i32 + dh;
            let vw = uw as i32 + dw;

            if vh < 0 || vh >= self.n as i32 {
                continue;
            }
            if vw < 0 || vw >= self.n as i32 {
                continue;
            }

            let vh = vh as usize;
            let vw = vw as usize;

            if self.c[vh][vw] == 0 {
                res.push([vh, vw]);
            }
        }
        res
    }

    fn greedy_move(&mut self, move_num: usize) -> Vec<[usize; 4]> {
        let mut rng = rand::thread_rng();

        let directions = vec![(0, 1), (-1, 0), (0, -1), (1, 0)];

        let mut cnt = 0;
        let mut move_list: Vec<[usize; 4]> = Vec::new();

        loop {
            cnt += 1;

            if cnt % 50 == 0 {
                let t = get_time();
                if t > 0.4 {
                    return move_list;
                }
            }

            let mut movables: Vec<[usize; 2]> = Vec::new();
            for h in 0..self.n {
                for w in 0..self.n {
                    if self.c[h][w] == 0 {
                        continue;
                    }
                    let movable_pos = self.get_movable_position([h, w]);

                    if movable_pos.len() > 0 {
                        movables.push([h, w]);
                    }
                }
            }
            movables.shuffle(&mut rng);

            for pc in movables {
                if self.c[pc[0]][pc[1]] == 0 {
                    continue;
                }

                let move_to = self.get_movable_position(pc);

                if move_to.len() == 0 {
                    continue;
                }

                for [vh, vw] in move_to {
                    if move_list.len() > move_num {
                        return move_list;
                    }
                    self.move_pc([pc[0], pc[1], vh, vw]);

                    let cables = self.get_all_cables();

                    if cables.len() <= self.cables.len() {
                        self.move_pc([vh, vw, pc[0], pc[1]]);
                    } else {
                        self.cables = cables;
                        move_list.push([pc[0], pc[1], vh, vw]);
                        cnt += 1;
                        break;
                    }
                }
            }
            if move_list.len() > move_num {
                return move_list;
            }
        }
    }

    fn anneal_move(&mut self) {}

    fn greedy_connection(&mut self) {
        self.connect_cables_randomly(self.k * 100);
        let time_limit: f64 = 2.5;

        loop {
            if get_time() / time_limit > 1.0 {
                break;
            }

            let u_score = self.calc_score();

            let cut = self.cut_cables_randomly(10);
            let connect = self.connect_cables_randomly(self.k * 100 - self.connected_cables.len());

            let v_score = self.calc_score();

            if v_score >= u_score {
                continue;
            }

            for cable in connect {
                self.cut_cable(cable);
            }

            for cable in cut {
                self.connect_cable(cable);
            }
        }
    }

    fn anneal_connection(&mut self, time_limit: f64, connection_num: usize) {
        self.connect_cables_randomly(connection_num);
        let mut rng = rand_pcg::Pcg64Mcg::new(890482);
        let TL: f64 = time_limit;
        let T0: f64 = self.calc_score() as f64;
        let T1: f64 = 2.0;
        let mut T = T0;

        let mut cut_connect_num = self.connected_cables.len() as f64;
        cut_connect_num *= 0.1;
        let cut_connect_num = cut_connect_num as usize;

        let mut cnt = 0;
        loop {
            cnt += 1;
            if cnt % 50 == 0 {
                let mut t = get_time();
                t /= TL;
                if t >= 1.0 {
                    self.connect_cables_randomly(connection_num - self.connected_cables.len());
                    return;
                }
                T = T0.powf(1.0 - t) * T1.powf(t);
            }

            let u_score = self.calc_score();

            let cut = self.cut_cables_randomly(cut_connect_num);
            let connect =
                self.connect_cables_randomly(connection_num - self.connected_cables.len());

            let v_score = self.calc_score();

            // println!(
            //     "loop: {} connection: {} u_score: {}  v_score: {} T: {}",
            //     cnt,
            //     self.connected_cables.len(),
            //     u_score,
            //     v_score,
            //     T
            // );

            if u_score > v_score && !rng.gen_bool(f64::exp((v_score as f64 - u_score as f64) / T)) {
                for cable in connect {
                    self.cut_cable(cable);
                }
                for cable in cut {
                    self.connect_cable(cable);
                }
            }
        }
    }
}

fn compute_score(n: i64, k: i64, c: Vec<Vec<i64>>, res: Answer) -> i64 {
    let c = c.to_vec();
    for &v in res.moves.iter() {
        let [i, j, i2, j2] = v;

        // 移動可能であるか確認
        if c[i][j] < 1 || c[i][j] > k {
            panic!(
                "passed position must be 1 <= and <= k but passed value is {}",
                c[i][j]
            );
        }
        if c[i2][j2] != 0 {
            panic!("passed position value must be 0 but {}", c[i2][j2]);
        }
    }

    let mut uf = Unionfind::new(k as usize);
    for &v in res.connects.iter() {
        let [i, j, i2, j2] = v;

        // 接続可能であるか確認
        if c[i][j] < 1 || c[i][j] > k {
            panic!(
                "passed position must be 1 <= and <= k but passed value is {}",
                c[i][j]
            );
        }
        if c[i2][j2] != 0 {
            panic!(
                "passed position must be 1 <= and <= k but passed value is {}",
                c[i2][j2]
            );
        }
        uf.unite((i, j), (i2, j2));
    }

    let mut computers: Vec<(usize, usize)> = Vec::new();
    for i in 0..n as usize {
        for j in 0..n as usize {
            if 1 <= c[i][j] && c[i][j] <= k {
                computers.push((i, j))
            }
        }
    }

    let mut score: i64 = 0;
    for i in 0..computers.len() {
        for j in i + 1..computers.len() {
            let c1 = computers[i];
            let c2 = computers[j];
            if uf.find(c1) != uf.find(c2) {
                continue;
            }
            if c[c1.0][c1.1] == c[c2.0][c2.1] {
                score += 1;
            } else {
                score -= 1;
            }
        }
    }
    std::cmp::max(score, 0)
}

fn main() {
    // 入力
    let mut s: String = String::new();
    std::io::stdin().read_line(&mut s).ok();

    let v: Vec<usize> = s
        .trim()
        .split_whitespace()
        .map(|e| e.parse::<usize>().ok().unwrap())
        .collect();
    let n: usize = v[0];
    let k: usize = v[1];

    let mut c: Vec<Vec<usize>> = Vec::new();
    for _ in 0..n {
        let mut s: String = String::new();
        std::io::stdin().read_line(&mut s).ok();
        let v: Vec<usize> = s
            .trim()
            .chars()
            .map(|e| e as usize - '0' as usize)
            .collect();
        c.push(v);
    }

    // pc 結合の焼きなまし
    get_time();
    let mut anneal = Anneal::new(&c, n, k);
    let moves = anneal.greedy_move(k * 30);

    anneal.anneal_connection(2.5, k * 100 - moves.len());

    let mut connects: Vec<[usize; 4]> = Vec::new();

    for &cable in anneal.connected_cables.iter() {
        connects.push(cable);
    }

    let ans = Answer::new(&moves, &connects);
    ans.print_answer();
    // println!("{}", anneal.calc_score());
}
