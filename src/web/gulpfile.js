'use strict';

// grab our gulp packages
var gulp  = require('gulp');
var $ = require('gulp-load-plugins')();
var browserSync = require('browser-sync');
var reload = browserSync.reload;

function handleError(err) {
  console.error(err.toString());
  this.emit('end');
}

gulp.task('images', function () {
  return gulp.src('build/images/*')
    .pipe($.cache($.imagemin({
      progressive: true,
      interlaced: true
    })))
    .pipe(gulp.dest('dist/images'));
});

gulp.task('scripts', function () {
  return gulp.src('build/scripts/*.js')
    .pipe($.changed('dist/scripts', {extension: '.js'}))
    .pipe($.plumber())
    .on('error', handleError)
    .pipe($.babel())
    .pipe(gulp.dest('dist/scripts'));
});

gulp.task('styles', function () {
  return gulp.src('build/styles/*.css')
    .pipe($.changed('dist/styles', {extension: '.css'}))
    .pipe($.plumber())
    .on('error', handleError)
    .pipe($.autoprefixer({browsers: ['last 1 version']}))
    .pipe(gulp.dest('dist/styles'))
    .pipe(reload({stream: true}));
});

gulp.task('html', ['images','scripts','styles'], function () {
  var assets = $.useref.assets({searchPath: ['build/*']});

  return gulp.src('build/*.html', {base: 'build/*'})
    .pipe(assets)
    .pipe($.if('*.js', $.uglify()))
    .pipe($.if('*.css', $.csso()))
    .pipe($.rev())
    .pipe(assets.restore())
    .pipe($.useref())
    .pipe($.revReplace())
    .pipe($.if('*.html', $.minifyHtml({conditionals: true, loose: true})))
    .pipe(gulp.dest('dist', { cwd: 'dist' }));
});


gulp.task('serve', ['html'], function() {
  browserSync({
    notify: false,
    port: 9000,
    server: {
      baseDir: ['build'],
    }
  });

  gulp.watch([
    'build/*.html',
    'build/scripts/*.js',
    'build/styles/*.css',
    'build/images/*'
  ]).on('change', reload);

});

gulp.task('build', ['html'], function () {
  return gulp.src('dist/*').pipe($.size({title: 'build'}));
});

gulp.task('default', function () {
  gulp.start('build');
});
