Coursera Machine Learining ノート #2 - Week 1: Model and Cost Function
========================================================

Model and Cost Function
--------------------------------------------------------
最初の学習アルゴリズムとして線形回帰(Linear regression)を学んでいく。

### Model Representation
- 今回も住宅価格のデータセットから
- この敷地面積から価格を予想するの例が
  Supervised LearningなのはRight Answerを事前に与えているから。ここ重要！
- データセットは別名で訓練セット(Training Set)とも呼ばれる
- シンボルの意味
    + \\( m \\) : 訓練サンプルの数
	+ \\( x's \\) : 入力変数 ( \\( 's \\) はsetの意味 )
	+ \\( y's \\) : 出力変数/ターゲット変数
	+ \\( (x, y) \\) : １件の訓練サンプル
    + \\( (x\^{(i)}, y\^{(i)}) \\) : i番目の訓練サンプル。
	  `()`で囲まれいる場合はインデックスを指す。i乗ではないので注意

- Learning AlgorithmはTraining Setを引数に受け取り、関数 \\( h \\) を出力する
- \\( h \\) は慣習的にそうしている。仮説(hypothesis)の略
- \\( h \\) に未知の入力変数を与えると、予測された出力変数を出力する
- So \\( h \\) is a function that maps from \\( x's \\) to \\( y's \\)

- 学習アルゴリズムを設計する際に、次に決めなければならないことは、
  どのようにこの仮説 \\( h \\) を表現するか (How do we represent h?)
- 最初は \\( h\_{\\theta}(x) = \\theta\_{0} + \\theta\_{1} x \\) としましょう
    + \\( h\_{\\theta}(x) \\) は省略して \\( h(x) \\) 書くこともある
- この \\( h\_{\\theta}(x) \\) は \\( x \\) の直線な関数(=線形関数)だと予測している
- Linear regression with one variable \\( x \\) => Univariate linear regression
- 次のビデオでこのモデルをどのように実装していくかを見ていく

### Cost Function
- Cost Function の訳は目的関数
- \\( h\_{\\theta}(x) = \\theta\_{0} + \\theta\_{1} x \\) の
  \\( \theta \\) をモデルのパラメータと言う
- Idea: Choose \\( \\theta\_{0} , \\theta\_{1} \\) so that \\( h\_{\\theta}(x) \\)
  is close to \\( y \\) for our training examples \\( x, y \\) (closeは近似させるの意味)
- これを形式化すると、\\( ( h\_{\\theta}(x) - y )\^2 \\) を最小化する
  \\( \\theta\_{0} , \\theta\_{1} \\) を求めるということになる
- \\( x , y \\) はデータセットなので、
  実際に最小化したいのは平均誤差 \\( \\sum\_{i=1}\^{m} ( h\_{\\theta}(x\^{(i)}) - y\^{(i)} )\^2 \\)
    +  \\( \\# \\) は訓練サンプルの「数」の省略表記で使う
- さらに平均化するために \\( \\frac{1}{2m} \\) を掛けて \\( \\frac{1}{2m} \\sum\_{i=1}\^{m} ( h\_{\\theta}(x\^{(i)}) - y\^{(i)} )\^2 \\)

これが最終的な目的関数、

\\(
  J( \\theta\_{0} , \\theta\_{1} ) = \\frac{1}{2m} \\sum\_{i=1}\^{m} ( h\_{\\theta}(x\^{(i)}) - y\^{(i)} )\^2
\\)

となる。これはSquare Error Functionとも呼ばれる。

次のビデオで目的関数 \\( J \\) を感覚的に掴んでいく。


### Cost Function - Intuition I


### Cost Function - Intuition II


--------------------------------------------------------

※ Coursera Machine Learining の目次は[こちら](/entry/coursera-ml/index)


<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
