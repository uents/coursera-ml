
```
% brew search octave
homebrew/science/octave                         Caskroom/cask/xoctave
```

```
% brew install octave
Error: No available formula with the name "octave"
==> Searching for similarly named formulae...
Error: No similarly named formulae found.
==> Searching taps...
These formulae were found in taps:
homebrew/science/octave                         Caskroom/cask/xoctave
To install one of them, run (for example):
  brew install homebrew/science/octave
```

```
% brew tap homebrew/science
==> Tapping homebrew/science
Cloning into '/usr/local/Library/Taps/homebrew/homebrew-science'...
remote: Counting objects: 596, done.
remote: Compressing objects: 100% (595/595), done.
remote: Total 596 (delta 1), reused 94 (delta 0), pack-reused 0
Receiving objects: 100% (596/596), 505.62 KiB | 148.00 KiB/s, done.
Resolving deltas: 100% (1/1), done.
Checking connectivity... done.
Tapped 585 formulae (616 files, 1.6M)
```


```
% brew install octave
==> Installing octave from homebrew/science
==> Tapping homebrew/x11
Cloning into '/usr/local/Library/Taps/homebrew/homebrew-x11'...
remote: Counting objects: 64, done.
remote: Compressing objects: 100% (64/64), done.
remote: Total 64 (delta 0), reused 7 (delta 0), pack-reused 0
Unpacking objects: 100% (64/64), done.
Checking connectivity... done.
Tapped 60 formulae (147 files, 159K)
octave: XQuartz is required to install this formula.
You can install with Homebrew Cask:
  brew install Caskroom/cask/xquartz
  
You can download from:
  https://xquartz.macosforge.org
  imake: XQuartz is required to install this formula.
  You can install with Homebrew Cask:
  brew install Caskroom/cask/xquartz
	  
You can download from:
  https://xquartz.macosforge.org
  transfig: XQuartz 2.7.2 is required to install this formula.
  You can install with Homebrew Cask:
  brew install Caskroom/cask/xquartz
		  
You can download from:
  https://xquartz.macosforge.org

Error: Unsatisfied requirements failed this build.
```

```
% brew search xquartz
Caskroom/cask/xquartz                           Caskroom/versions/xquartz-beta
```

```
% brew cask install xquartz
==> Downloading https://dl.bintray.com/xquartz/downloads/XQuartz-2.7.9.dmg
######################################################################## 100.0%
==> Verifying checksum for Cask xquartz
==> Running installer for xquartz; your password may be necessary.
==> Package installers may write to any location; options such as --appdir are ignored.
Password:
==> installer: Package name is XQuartz 2.7.9
==> installer: Installing at base path /
==> installer: The install was successful.
🍺  xquartz staged at '/opt/homebrew-cask/Caskroom/xquartz/2.7.9' (73M)
```

```
% brew install octave

(snip...)

The graphical user interface is now used when running Octave interactively.
The start-up option --no-gui will run the familiar command line interface.
The option --no-gui-libs runs a minimalist command line interface that does not
link with the Qt libraries and uses the fltk toolkit for plotting if available.


Gnuplot is configured as default graphics toolkit, this can be changed within
Octave using 'graphics_toolkit'. Other Gnuplot terminals can be used by setting
the environment variable GNUTERM and building gnuplot with the following options.

  setenv('GNUTERM','qt')    # Requires QT; install gnuplot --with-qt
  setenv('GNUTERM','x11')   # Requires XQuartz; install gnuplot --with-x11
  setenv('GNUTERM','wxt')   # Requires wxmac; install gnuplot --with-wxmac
  setenv('GNUTERM','aqua')  # Requires AquaTerm; install gnuplot --with-aquaterm
		
You may also set this variable from within Octave. For printing the cairo backend
is recommended, i.e., install gnuplot with --with-cairo, and use
		
  print -dpdfcairo figure.pdf
		  
		  
When using the native qt or fltk toolkits then invisible figures do not work because
osmesa does currently not work with the Mac's OpenGL implementation. The usage of
gnuplot is recommended.
		  
		  
Octave has been compiled with Apple's BLAS routines, this leads to segfaults in some
tests. The option "--with-openblas" is a more conservative choice.
		  
		  
==> Summary
🍺  /usr/local/Cellar/octave/4.0.2_3: 2,202 files, 50.6M
```

```
% octave
dyld: Library not loaded: /usr/local/opt/fontconfig/lib/libfontconfig.1.dylib
  Referenced from: /usr/local/Cellar/octave/4.0.2_3/libexec/octave/4.0.2/exec/x86_64-apple-darwin14.5.0/octave-gui
  Reason: image not found
octave exited with signal 5
```
