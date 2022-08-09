#[cfg(feature = "local")]
use log::{debug, error, info, warn, LevelFilter};
use std;

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


fn main() {

    // 入力
    let mut s: String = String::new();
    std::io::stdin().read_line(&mut s).ok();

    let v: Vec<i64> = s.trim().split_whitespace().map(|e| e.parse::<i64>().ok().unwrap()).collect();
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
        for j in 0..n{
            let j : usize = j as usize;
            print!("{}", c[i][j])
        }
        print!("\n")
    }




}
