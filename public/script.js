$(function(){
    //複数のcanvas<canvas>タグを取得
    var $canvas = $('canvas')
    //そのうちの最初の要素を取得
    var canvas = $canvas.get(0)
    //2Dコンテキストオブジェクトを作る
    var context = canvas.getContext('2d')
    // canvasのイメージに対するdata: URLを返します。デフォルトはimage/png
    var blank = canvas.toDataURL()

    // 18のペン先で
    context.lineWidth = 18
    //接続は丸い
    context.lineJoin = 'round'

    //$.fnはjQueryオブジェクトのコンストラクタのprotoyypeプロパティ
    //こうすることで全てのjQueryオブジェクトにdrawTouchメソッドを生やすことができるらしい
    //http://q.hatena.ne.jp/1359552909
    $.fn.drawTouch = function() {
        var $this = $(this)
        var start = function (e) {
            //ブラウザの持っている機能を抑制する
            e.preventDefault()
            //タッチイベント発生時の座標をひろう
            var touchEvent = e.originalEvent.changedTouches[0]
            //現在のパスをリセットする
            context.beginPath()
            //左上橋からの位置
            context.moveTo(touchEvent.pageX - $this.offset().left, touchEvent.pageY - $this.offset().top)
        }

        var move = function (e) {
            e.preventDefault()
            var touchEvent = e.originalEvent.changedTouches[0]
            context.lineTo(touchEvent.pageX - $this.offset().left, touchEvent.pageY - $this.offset().top)
            context.stroke()
        }

        //第一引数はなんでも良さそう
        $this.on('touchstart', start)
        $this.on('touchmove', move)
    }

    $.fn.drawMouse = function(){
        var $this = $(this)
        var clicked = 0
        var start = function(e) {
            clicked = 1
            context.beginPath()
            context.moveTo(e.pageX - $this.offset().left, e.pageY - $this.offset().top)
        }
        var move = function(e) {
            if(clicked){
                context.lineTo(e.pageX - $this.offset().left, e.pageY - $this.offset().top)
                context.stroke()
            }
        }
        var stop = function(e) {
            clicked = 0
        }
        $this.mousedown(start)
        $this.mousemove(move)
        $this.mouseup(stop)
    }
    $canvas.drawTouch
    $canvas.drawMouse

    var $table = $('table')
    var clearCanvas = function(){
        context.clearRect(0, 0, canvas.width, canvas.height)
    }

    $submit = $('.js-submit')
    $submit.click(function(e){
        var $this = $(this)
        var dataURL = canvas.toDataURL()
        if(blank == dataURL) {
            return
        }
        //thisを無効にする
        $this.prop('disabled', true)
        $.post('/predict', { data_url: dataURL }, function(json) {
            //thisを有効にする
            $this.prop('disabled', false)
            var $img = $('<img>').prop('src', json.image)
            var $tr = $('<tr>')
            //<tr>
            //   <td><img src="hoge" ></img></td>
            //</tr>
            $('<td>').html($img).appendTo($tr)
            $('<td class="label">').html(json.label).appendTo($tr)
            $('<td>').html(json.percent + '%').appendTo($tr)
            $tr.prependTo($table).hide().fadeIn()
            clearCanvas()
        }, 'json')
    })

    $('.js-clear').click(function(e){
        clearCanvas()
        $submit.prop('disabled', false)
        $table.html('')

    })


})
