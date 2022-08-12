#[allow(unused_imports)]
#[cfg(feature = "local")]
use log::{debug, error, info, warn, LevelFilter};

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
        if self.parents[x.0 * self.size + x.1] < 0 {
            return x.0 * self.size + x.1;
        } else {
            return self.find(x);
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

struct Status {
    moves: Vec<[usize; 4]>,
    connects: Vec<[usize; 4]>,
    c: Vec<Vec<usize>>,
    n: usize,
    lim: usize, // k
}

impl Status {
    pub fn new(c: &Vec<Vec<usize>>, n: usize, lim: usize) -> Self {
        Status {
            moves: Vec::new(),
            connects: Vec::new(),
            c: c.clone(),
            n,
            lim,
        }
    }

    pub fn solve(&mut self) {
        self._move();
        self.connect();
    }
    fn _move(&mut self) {
        let move_direction: [[i64; 2]; 4] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        let move_limit = self.lim / 2;
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
    fn connect(&mut self) {
        let used_cmputer: usize = 9;
        let connect_limit: usize = self.lim / 2;

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

    let mut status = Status::new(&c, n, k * 100);
    status.solve();

    let ans = Answer::new(&status.moves, &status.connects);
    ans.print_answer();
}
