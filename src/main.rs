#[cfg(feature = "local")]
use log::{debug, error, info, warn, LevelFilter};

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
}

impl Unionfind {
    fn new(n: usize) -> Self {
        Unionfind {
            parents: vec![-1; n], // 負値:: -size  正値:: parent を示す
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parents[x] < 0 {
            return x;
        } else {
            return self.find(x);
        }
    }

    fn unitre(&mut self, x: usize, y: usize) {
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

struct Result {
    moves: Vec<(usize, usize, usize, usize)>,
    connects: Vec<(usize, usize, usize, usize)>,
}

fn compute_score(n: i64, k: i64, c: Vec<Vec<i64>>, res: Result) {
    let mut c = c.to_vec();
    for v in res.moves.iter() {
        let (i, j, i2, j2) = v;
    }
}

fn main() {
    // 入力
    let mut s: String = String::new();
    std::io::stdin().read_line(&mut s).ok();

    let v: Vec<i64> = s
        .trim()
        .split_whitespace()
        .map(|e| e.parse::<i64>().ok().unwrap())
        .collect();
    let n: i64 = v[0];
    let k: i64 = v[1];

    print!("{} {}", n, k);

    let mut c: Vec<Vec<i64>> = Vec::new();
    for _ in 0..n {
        let mut s: String = String::new();
        std::io::stdin().read_line(&mut s).ok();
        let v: Vec<i64> = s.trim().chars().map(|e| e as i64 - '0' as i64).collect();
        c.push(v);
    }

    for i in 0..n {
        let i: usize = i as usize;
        for j in 0..n {
            let j: usize = j as usize;
            print!("{}", c[i][j])
        }
        print!("\n")
    }
}
