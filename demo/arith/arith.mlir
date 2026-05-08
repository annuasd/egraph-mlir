module {
  func.func @arith_demo(%x: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %mul = arith.muli %x, %c2 : i32
    %div = arith.divsi %mul, %c2 : i32
    return %div : i32
  }
}
