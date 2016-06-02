Coursera Machine Learining ノート #1 - Week 1: Introduction
========================================================

始まりました。英語で大苦戦してます。。

Welcome
--------------------------------------------------------

### Welcome to Machine Learning! (動画)
Machine Learningとは何か？ 色んな所で使われているよね。

- GoogleやBingのページランクを行うソフトウェア
- FacebookやApple Photo applicationでの写真の中の友達認識
- Emailクライアントのスパムフィルタ

つまりMachine Learningとは、明示的にプログラムされないなくても、
学習することができるコンピュータを獲得する科学と言えます。

(中略...)

私にとって、Machine Leariningを学ぶ理由の一つは、それがAIでありエキサイティングで、
あなたや私のように知性のある機械を構築することです。
たくさんの科学者はこれらを学習を通して進化させることが最もよい方法だと考えています。
そこで、脳の働きを模倣したニューラルネットワークと呼ばれるアルゴリズムを教えて行きます。

このクラスでは、Machine Leariningについて学び、自らそれらを実装することが目標です。
ここにサインアップして私たちのクラスに入ってくれることを願っています。


### Machine Learning Honor Code
Honor Codeとは「規定」とか「規約」といった意味らしい。

- 生徒たちによるグループワークを強く推奨
- 宿題としてreview questionsを提出すること
- programming execisesはアルゴリズムの目的など他の生徒とのよい議論になる
- 他の生徒が書いたソースコードは見ないように、そして答えを他の生徒には見せないように

### Guidelines for Posting Code in Discussion Forums
省略。


Introduction
--------------------------------------------------------

### Welcome (動画)
ここから本題の講義へ。ここの動画には日本語の字幕がありました。

- 冒頭はこれまでの内容の振り返り
- 機械学習を学んでも実際に関心のある問題に適用できないと意味がないので、
  その方法も学んでいくよ。そのために演習問題をかなりの時間をかけて開発しました
  
- 機械学習の応用例
    + データベース・マインニング
        * ウェブページのクリックのデータ
	    * 電子カルテ。医療データから医療知識の抽出
	    * 計算生物学。遺伝子配列やDNA配列のデータ解析
		* 工学。増加していく技術データに対して学習による理解
    + 手作業でプログラム化できない分野
	    * ヘリコプターの自律制御
		* 手書き認識。例えば郵便の自動仕分け
		* 自然言語処理(NLP)
		* コンピュータビジョン
    + Self-costomizing Programs
	    * product recommendation。個人の嗜好に合わせた自己カスタマイズ
    + Understainding human learning (brain, real AI)


### How to Use Discussion Forums
Discussion Forumの使い方について。

誹謗中傷やスパム行為のような行動は止めてね、みたいなことの説明や、
投稿手順について。まあ使う時が来たらちゃんと読もう。

### Supervised Learining (動画)
教師あり学習(Supervised Learning)について。

- 住宅価格の予測の例。敷地面積と価格のいくつかのデータセットがあって
  どう予測するか 
- 教師あり学習とは、データセットを正しい答えして与え、
  未知のパラメータをどう予測さえるか
- より専門的に言うと回帰問題(Regression)

- 別の例として、カルテの情報を見て乳がんが悪性か良性かの予測
- 腫瘍のサイズで悪性か良性かの見分けがつくか
- より専門的に言うと分類問題(Classification)
- ここでSVMの紹介が少し

最後にクイズで、いくつかの例で回帰問題or分類問題のどちら？かを尋ねられる。


### Unsupervised Learining (動画)
教師なし学習(Unsupervised Learning)について。

- クラスタリング：ラベルのないデータを異なるクラスに分類すること
- 応用例
    + Google Newsの記事分類
	+ ゲノム解析による個人の異なるタイプ・カテゴリの分類
	+ データセンターのコンピュータクラスタリング
	+ SNSのネットワーク解析
	+ マーケット・セグメンテーション
	+ 天文学のデータ解析
- カクテルパーティー
    + 2人の声が混ざった音源を別々に分類する
	+ Octaveなら一行で書けるよ！
	    * 例えば、行列の得異値分解を`svd()`だけで行うことができる
        * Octaveなら機械学習を非常に早く学習しプロトタイピングできる
		
最後にクイズで、いくつかの例で教師あり学習or教師なし学習のどちら？かを尋ねられる。


### Who are Mentors?
CourseraにはMentorにふさわしい人がたくさんいあるから、
discussion areaに自分のアイデアをどんどん出して行って、とかそんな内容。

### Get to Know Your Classmates
クラスメートと一緒にやるのがオンラインコースの重要店なので、
コースの早いうちから打ち解けてお互いを知りましょう。
(「打ち解ける」は`break the ice`って言うのか)

自己紹介でオススメのトピックは、

- Where are you from?
- Career and education?
- Hopes?
- Other into? 

らしい。なんか昔通っていた英会話教室のレッスンみたいだ。

あと、プロフィールのページもあるから更新してね、とか。

### Frequently Asked Questions
このコースを受けるための前提条件(必要なスキル)とか、
なんでOctaveやねんとか、ビデオってダウンロードできないのとか。

まあ、僕の場合はそもそも「英語」が最大の難関ですが...


Review
--------------------------------------------------------

### Quiz
パスするには Earn at least 80%

1回目は正答率3/5で不合格。2回目は4/5でパス。

たぶんこのクイズの内容より、英語が難しくて読み間違えてる気がする。

パスした時は、間違った問題の答えを教えてほしいんだけど、
そうじゃないので辛い。


Other Materials
--------------------------------------------------------

### Machine Learning Course Wiki
Wikiもあるのでご参考に、といった内容。

次回は、Week1の続き Linear Regression with One Variable から。


--------------------------------------------------------

※Coursera Machine Learining の目次は[こちら](/entry/coursera-ml/index)


<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: { inlineMath: [['$','$'], ["\\(","\\)"]] } });
</script>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML">
</script>
<meta http-equiv="X-UA-Compatible" CONTENT="IE=EmulateIE7" />
