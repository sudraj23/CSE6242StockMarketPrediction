/* global d3, _ */
	
(function() {
	 d3.json("/static/asset.json", function (states) {
        //console.log(states.items)
        var colorScale = d3.scale.category20();

        var scrollSVG = d3.select(".viewport").append("svg")
            .attr("class", "scroll-svg")
            .attr("transform","translate(" + 0 + "," + (0) + ")")
            ;

        var defs = scrollSVG.insert("defs", ":first-child");

        createFilters(defs);

        var chartGroup = scrollSVG.append("g")
            .attr("class", "chartGroup")
            //.attr("filter", "url(#dropShadow1)"); // sometimes causes issues in chrome

        chartGroup.append("rect")
            .attr("fill", "#FFFFFF");

        var infoSVG = d3.select(".information").append("svg")
            .attr("class", "info-svg");

        var rowEnter = function(rowSelection) {
            rowSelection.append("rect")
                .attr("rx", 3)
                .attr("ry", 3)
                .attr("width", "280")
                //.on("click",fMouseClick)
                //.on("mouseover",fMouseOver)
                .attr("height", "24")
                .attr("fill-opacity", 0.6)
                .attr("stroke", "#999999")
                .attr("stroke-width", "2px");
            rowSelection.append("text")
                .attr("transform", "translate(10,15)");
                rowSelection.on("click",fMouseClick)
                .on("mouseover",fMouseOver);
        };

        var rowUpdate = function(rowSelection) {
            rowSelection.select("rect")
                .attr("fill", '#E7E7E7');//function(d) { //console.log(d.label);
                    //return colorScale("Morgan Stanley");//change id to label
                //})
                //.on("click",fMouseClick);
            rowSelection.select("text")
                .text(function (d) {
                    return (d.index + 1) + ". " + d.label;
                }).attr('font-weight', 600);
        };

        var rowExit = function(rowSelection) {
        };

        var virtualScroller = d3.VirtualScroller()
            .rowHeight(30)
            .enter(rowEnter)
            .update(rowUpdate)
            .exit(rowExit)
            .svg(scrollSVG)
            .totalRows(1001)
            .viewport(d3.select(".viewport"));

        // tack on index to each data item for easy to read display
        states.items.forEach(function(nextState, i) {
            nextState.index = i;
        });
        var maxim = d3.max(states.items.map(function(d) { return d.index; }));
        console.log(maxim)
        virtualScroller.data(states.items, function(d) { return d.label; });//change id to label

        chartGroup.call(virtualScroller);

        function createFilters(svgDefs) {
            var filter = svgDefs.append("svg:filter")
                .attr("id", "dropShadow1")
                .attr("x", "0")
                .attr("y", "0")
                .attr("width", "200%")
                .attr("height", "200%");

            filter.append("svg:feOffset")
                .attr("result", "offOut")
                .attr("in", "SourceAlpha")
                .attr("dx", "1")
                .attr("dy", "1");

            filter.append("svg:feColorMatrix")
                .attr("result", "matrixOut")
                .attr("in", "offOut")
                .attr("type", "matrix")
                .attr("values", "0.1 0 0 0 0 0 0.1 0 0 0 0 0 0.1 0 0 0 0 0 0.2 0");

            filter.append("svg:feGaussianBlur")
                .attr("result", "blurOut")
                .attr("in", "matrixOut")
                .attr("stdDeviation", "1");

            filter.append("svg:feBlend")
                .attr("in", "SourceGraphic")
                .attr("in2", "blurOut")
                .attr("mode", "normal");
        }
    
 var margin = {top: 30, right: 20, bottom: 100, left: 50},
    margin2  = {top: 210, right: 20, bottom: 20, left: 50},
    width    = 764 - margin.left - margin.right+200,
    height   = 283 - margin.top - margin.bottom+100,
    height2  = 283 - margin2.top - margin2.bottom;

 function fMouseOver(d){
 	d3.select(this).style("cursor", "pointer");
 }

 var svg = d3.select('body').append('svg')
    .attr('class', 'chart')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom+60+120)
     .attr("transform","translate(" + 100 + "," + (10) + ")");
var scrollSVG1 = d3.select(".viewport1").append("svg")
            .attr("class", "scroll-svg1")
            .attr("transform","translate(" + 0 + "," + (0) + ")")
            ;
var infoSVG1 = d3.select(".information1").append("svg")
            .attr("class", "info-svg");

var svgone = d3.select(".information3").append("svg")
             .attr("class",'chart')
             .attr("transform","translate(" + 0 + "," + (0) + ")")
            .attr("width", 280)
            .attr("height", 50);
var legend1 = svgone.append('g')
    .attr('class', 'chart__legend')
    .attr('width', width)
    .attr('height', 50)
    .attr('transform', 'translate(' + (0) + ', 10)')
    .style('font-size',15);
  

  legend1.append('text')
    .attr('class', 'chart__legend')
    .attr('transform', 'translate(' + (70)  + ',' + (20) + ')')
    .style('fill','black')
    .style("font-weight", "bold")
    .text('List of Stocks');
var div = d3.select("body").append("div") 
    .attr("class", "tooltip")       
    .style("opacity", 0.9);

            //.style("float","up");
            // .attr("class","bar-chart")
            // .append("g")
 var count=0;
 fMouseClick({"label": ["Morgan Stanley"]})

 function fMouseClick(d){
  	//console.log(d.label)
  if (count!=0){
  d3.selectAll('rect').attr({class:'bar'});
  d3.select(this).select('rect').attr({class:"bar1"});
  }
  count=count+1;
  svg.selectAll("*").remove();
  scrollSVG1.selectAll("*").remove();
  infoSVG1.selectAll("*").remove();
  var parseDate = d3.time.format('%d/%m/%Y').parse,
    bisectDate = d3.bisector(function(d) { return d.date; }).left,
    legendFormat = d3.time.format('%b %d, %Y');

  var x = d3.time.scale().range([0, width]),
    x2  = d3.time.scale().range([0, width]),
    y   = d3.scale.linear().range([height, 0]),
    y1  = d3.scale.linear().range([height, 0]),
    y2  = d3.scale.linear().range([40, 0]),
    y3  = d3.scale.linear().range([40, 0]);

  var xAxis = d3.svg.axis().scale(x).orient('bottom'),
    xAxis2  = d3.svg.axis().scale(x2).orient('bottom'),
    yAxis   = d3.svg.axis().scale(y).orient('left');

  var priceLine = d3.svg.line()
    .interpolate('monotone')
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.price); });

  var avgLine = d3.svg.line()
    .interpolate('monotone')
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.average); });

  var area2 = d3.svg.area()
    .interpolate('monotone')
    .x(function(d) { return x2(d.date); })
    .y0(height2)
    .y1(function(d) { return y2(d.price); });

  
  svg.append('defs').append('clipPath')
    .attr('id', 'clip')
  .append('rect')
    .attr('width', width)
    .attr('height', height);

  var make_y_axis = function () {
    return d3.svg.axis()
      .scale(y)
      .orient('left')
      .ticks(3);
  };
var make_x_axis = function () {
    return d3.svg.axis()
      .scale(x)
      .orient('bottom')
      .ticks(3);
  };


  var focus = svg.append('g')
    .attr('class', 'focus')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  var barsGroup = svg.append('g')
    .attr('class', 'volume')
    .attr('clip-path', 'url(#clip)')
    .attr('transform', 'translate(' + margin.left + ',' + (margin.top+40+120) + ')');

  var context = svg.append('g')
    .attr('class', 'context')
    .attr('transform', 'translate(' + (margin2.left) + ',' + (margin2.top+40+100 ) + ')');

  var legend = svg.append('g')
    .attr('class', 'chart__legend')
    .attr('width', width)
    .attr('height', 500)
    .attr('transform', 'translate(' + (margin2.left + 200) + ', 10)')
    .style('font-size',15);
  

  legend.append('text')
    .attr('class', 'chart__legend')
    .attr('transform', 'translate(' + (- 200)  + ',' + (10) + ')')
    .style('fill','#313b4e')
    .style("font-weight", "bold")
    .text('Asset: '+d.label);

 var legend1 = svg.append('g')
    .attr('class', 'chart__legend')
    .attr('width', width)
    .attr('height', 500)
    .attr('transform', 'translate(' + (margin2.left) + ', 50)')
    .style('font-size',15);
  

  legend.append('text')
    .attr('class', 'chart__legend')
    .attr('transform', 'translate(' + (10)  + ',' + (420) + ')')
    .style('fill','black')
    .style("font-weight", "bold")
    .text('News Headlines for the Selected Date for the Selected Stock');

    
  var rangeSelection =  legend
    .append('g')
    .attr('class', 'chart__range-selection')
    .attr('transform', 'translate(110, 0)');


  
  url='/asset?assetName='+(d.label).toString().replace(/[&/]/g,"")+'&start_time=2007-01-01&end_time=2016-12-31';
  assetname=d.label
  console.log(d.label)
  console.log((d.label).toString().replace(/[&/]/g,""))
  //'/static/data/aapl.csv'
  d3.csv(url, type, function(err, data,assetName) {
    var brush = d3.svg.brush()
      .x(x2)
      .on('brush', brushed);

    var xRange = d3.extent(data.map(function(d) { return d.date; }));

    x.domain(xRange);
    y.domain(d3.extent(data.map(function(d) { return d.price; })));
    y3.domain(d3.extent(data.map(function(d) { return d.price; })));
    x2.domain(x.domain());
    y2.domain(y.domain());

    var min = d3.min(data.map(function(d) { return d.price; }));
    var max = d3.max(data.map(function(d) { return d.price; }));

    // var range = legend.append('text')
    //   .text(legendFormat(new Date(xRange[0])) + ' - ' + legendFormat(new Date(xRange[1])))
    //   .style('text-anchor', 'end')
    //   .attr('transform', 'translate(' + width + ', 0)');

    focus.append('g')
        .attr('class', 'y chart__grid')
        .call(make_y_axis()
        .tickSize(-width, 0, 0)
        .tickFormat(''));

    focus.append('g')
        .attr('class', 'x chart__grid')
        .call(make_x_axis()
        .tickSize(254,0 , 0)
        .tickFormat(''));

    var averageChart = focus.append('path')
        .datum(data)
        .attr('class', 'chart__line chart__average--focus line')
        .attr('d', avgLine);

    var priceChart = focus.append('path')
        .datum(data)
        .attr('class', 'chart__line chart__price--focus line')
        .attr('d', priceLine);

    focus.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0 ,' + height + ')')
        .call(xAxis);

    focus.append('g')
        .attr('class', 'y axis')
        .attr('transform', 'translate(0, 0)')
        .call(yAxis);

    var focusGraph = barsGroup.selectAll('rect')
        .data(data)
      .enter().append('rect')
        .attr('class', 'chart__bars')
        .attr('x', function(d, i) { return x(d.date); })
        .attr('y', function(d) { return 155 - y3(d.price); })
        .attr('width', 1)
        .attr('height', function(d) { return y3(d.price); });

    var helper = focus.append('g')
      .attr('class', 'chart__helper')
      .style('text-anchor', 'end')
      .attr('transform', 'translate(' + (-730+width) + ', 5)');

    var helperText = helper.append('text')
                     .style('font-weight','bold')
                     .style('fill','grey')

    var priceTooltip = focus.append('g')
      .attr('class', 'chart__tooltip--price')
      .append('circle')
      .style('display', 'none')
      .attr('r', 2.5);

    var averageTooltip = focus.append('g')
      .attr('class', 'chart__tooltip--average')
      .append('circle')
      .style('display', 'none')
      .attr('r', 2.5);

    var mouseArea = svg.append('g')
      .attr('class', 'chart__mouse')
      .append('rect')
      .attr('class', 'chart__overlay')
      .attr('width', width)
      .attr('height', height)
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
      .on('click',ffmouseclick)
      .on('mouseover', function() {
      // var x0 = x.invert(d3.mouse(this)[0]);
      // var i = bisectDate(data, x0, 1);
      // var d0 = data[i - 1];
      // var d1 = data[i];
      // var d = x0 - d0.date > d1.date - x0 ? d1 : d0;
      // var formattime=d3.time.format('%Y-%m-%d')
        helper.style('display', null);
        priceTooltip.style('display', null);
        averageTooltip.style('display', null);
       
      })
      .on('mouseout', function() {
        helper.style('display', 'none');
        priceTooltip.style('display', 'none');
        averageTooltip.style('display', 'none');
        div.transition()    
                .duration(500)    
                .style("opacity", 0); 
      })
      .on('mousemove', mousemove);
      
    context.append('path')
        .datum(data)
        .attr('class', 'chart__area area')
        .attr('d', area2);

    context.append('g')
        .attr('class', 'x axis chart__axis--context')
        .attr('y', 0)
        .attr('transform', 'translate(0,' + (height2 - 22) + ')')
        .call(xAxis2);

    context.append('g')
        .attr('class', 'x brush')
        .call(brush)
      .selectAll('rect')
        .attr('y', -6)
        .attr('height', height2+7 );

    function mousemove() {
      var x0 = x.invert(d3.mouse(this)[0]);
      var i = bisectDate(data, x0, 1);
      var d0 = data[i - 1];
      var d1 = data[i];
      var d = x0 - d0.date > d1.date - x0 ? d1 : d0;
      var formattime=d3.time.format('%Y-%m-%d')
      console.log(d.date)
      if (d.Expected==1 && d.Predicted==1){
        
        var ImagePath = "/static/im1.jpg"; // you will have to set the image path here
      }else if(d.Expected==1 && d.Predicted==0){
        var ImagePath = "/static/im3.jpg";
      }else if(d.Expected==0 && d.Predicted==0){
        var ImagePath = "/static/im2.jpg";
      }else if(d.Expected==0 && d.Predicted==1){
        var ImagePath = "/static/im4.jpg"; 
      }  

      helper.append('rect')
         .attr('transform', 'translate(' + (- 160)  + ',' + (0) + ')')
         .attr('width', 90)
         .attr('height', 52)
         .style('fill','#ffffff')
         .style('opacity', 1)

     helper.append('text')
           .attr('dx','-9em')
           .attr('dy', '1em')
           .style('font-weight','bold')
           .style('fill','black')
           .text(legendFormat(new Date(d.date)))

     helper.append('text')
           .attr('dx','-9em')
           .attr('dy', '3em')
           .style('font-weight','bold')
           .style('fill','black')
           .style('text-anchor', 'left')
           .text('Price: ' + Math.round(d.price*100)/100)

     helper.append('text')
           .attr('dx','-9em')
           .attr('dy', '5em')
           .style('font-weight','bold')
           .style('fill','black')
           .style('text-anchor', 'left')      
           .text('Avg: ' + Math.round(d.average*100)/100)

      // helperText.text(legendFormat(new Date(d.date)) + ' - Price: ' + Math.round(d.price*100)/100 + ' Avg: ' + Math.round(d.average*100)/100);
      priceTooltip.attr('transform', 'translate(' + x(d.date) + ',' + y(d.price) + ')');
      averageTooltip.attr('transform', 'translate(' + x(d.date) + ',' + y(d.average) + ')');
       // div.transition()    
       //          .duration(200)    
       //          .style("opacity", .9);    
       //  div.html("Actual Trend: " + actual + "<br/>"  + "Model Prediction: "+ " ")  
       //          .style("left", (d3.event.pageX) + "px")   
       //          .style("top", (d3.event.pageY - 75) + "px");
      var string = "<img src="+ ImagePath+ "/>";
      div .html(string) //this will add the image on mouseover
      .attr('width', 100)
      .attr('height', 50)
      .style("left", (d3.event.pageX + 10) + "px")     
      .style("top", (d3.event.pageY -50) + "px")
      .style('opacity',0.9)
      
                    //.style("font-color", "blue");

       
    }

    function brushed() {
      var ext = brush.extent();
      if (!brush.empty()) {
        x.domain(brush.empty() ? x2.domain() : brush.extent());
        y.domain([
          d3.min(data.map(function(d) { return (d.date >= ext[0] && d.date <= ext[1]) ? d.price : max; })),
          d3.max(data.map(function(d) { return (d.date >= ext[0] && d.date <= ext[1]) ? d.price : min; }))
        ]);
        //range.text(legendFormat(new Date(ext[0])) + ' - ' + legendFormat(new Date(ext[1])))
        focusGraph.attr('x', function(d, i) { return x(d.date); });

        var days = Math.ceil((ext[1] - ext[0]) / (24 * 3600 * 1000))
        focusGraph.attr('width', (40 > days) ? (40 - days) * 5 / 6 : 5)
      }

      priceChart.attr('d', priceLine);
      averageChart.attr('d', avgLine);
      focus.select('.x.axis').call(xAxis);
      focus.select('.y.axis').call(yAxis);
    }

    var dateRange = ['1w', '1m', '3m', '6m', '1y', '5y']
    for (var i = 0, l = dateRange.length; i < l; i ++) {
      var v = dateRange[i];
      rangeSelection
        .append('text')
        .attr('class', 'chart__range-selection')
        .text(v)
        .style("fill", "#313b4e")
        .style("font-weight", "bold")
        .attr('transform', 'translate(' + (30 * i) + ', 5)')
        .on('click', function(d) { focusOnRange(this.textContent); });
    }

    function focusOnRange(range) {
      var today = new Date(data[data.length - 1].date)
      var ext = new Date(data[data.length - 1].date)

      if (range === '1m')
        ext.setMonth(ext.getMonth() - 1)

      if (range === '1w')
        ext.setDate(ext.getDate() - 7)

      if (range === '3m')
        ext.setMonth(ext.getMonth() - 3)

      if (range === '6m')
        ext.setMonth(ext.getMonth() - 6)

      if (range === '1y')
        ext.setFullYear(ext.getFullYear() - 1)

      if (range === '5y')
        ext.setFullYear(ext.getFullYear() - 5)

      brush.extent([ext, today])
      brushed()
      context.select('g.x.brush').call(brush.extent([ext, today]))
    }

// Function for news data extraction


function ffmouseclick(){
var x0 = x.invert(d3.mouse(this)[0]);
scrollSVG1.selectAll("*").remove();
infoSVG1.selectAll("*").remove();
// svgone.selectAll("*").remove();
var i = bisectDate(data, x0, 1);
var d0 = data[i - 1];
var d1 = data[i];
var d = x0 - d0.date > d1.date - x0 ? d1 : d0;
var formattime=d3.time.format('%Y-%m-%d')
url='/news?assetName='+assetname[0].toString().replace(/&/g,"")+'&start_time='+formattime(d.date)+'&end_time='+formattime(d3.time.day.offset(d.date,0));
// var barChart = svgone.selectAll("rect")
//     .data([150])
//     .enter()
//     .append("rect")
//     .attr("y", function(d) {
//         return 200 - d
//     })
//     .attr('x',50)
//     .attr("height", function(d) {
//         return d;
//     })
//     .attr("width", 70);
    // .attr("transform", function (d, i) {
    //      var translate = [barWidth * i, 0];
    //      return "translate("+ translate +")";
    // });


// if (x0 - d0.date > d1.date - x0){
// if(data[i+10].price>=data[i].price){
//   console.log("Hell");

//    barChart.attr("class",'bar-chart1');
// }
// else{
//   console.log("Hell No");
  
//    barChart.attr("class",'bar-chart2');
// }
// }else{
  
//   if(data[i+9].price<data[i].price){
//   console.log("Heaven")
//   barChart.attr("class",'bar-chart1');
//   }else{
//   console.log("Heaven No")
//    barChart.attr("class",'bar-chart2');
// }
// }



d3.select(this).style("cursor", "pointer");
console.log(url)
d3.json(url, function (states) {
        //console.log(states.items)
        //var colorScale = d3.scale.category20();
        var colorScale = d3.scale.ordinal()
        .domain(['0','1'])
        .range(['red','green']);
        // // var scrollSVG1 = d3.select(".viewport1").append("svg")
        //     .attr("class", "scroll-svg")
        //     .attr("transform","translate(" + 0 + "," + (0) + ")")
        //     ;

        var defs = scrollSVG1.insert("defs", ":first-child");
        console.log(states)
        states.forEach(function(nextState, i) {
            nextState.index = i;
         });

        var maxim = d3.max(states.map(function(d) { return d.index; }));
        if (states.length==0){
          states=[{'date':'','label':'No news headline Found for this search'}];
          maxim=1;
        }
        createFilters(defs);

        var chartGroup = scrollSVG1.append("g")
            .attr("class", "chartGroup")
            //.attr("filter", "url(#dropShadow1)"); // sometimes causes issues in chrome

        chartGroup.append("rect")
            .attr("fill", "#FFFFFF");

        // var infoSVG1 = d3.select(".information1").append("svg")
        //     .attr("class", "info-svg");

        var rowEnter = function(rowSelection) {
            rowSelection.append("rect")
                .attr("rx", 3)
                .attr("ry", 3)
                .attr("width", "1050")
                //.on("click",fMouseClick)
                //.on("mouseover",fMouseOver)
                .attr("height", "30")
                .attr("fill-opacity", 0.6)
                .attr("stroke", "#999999")
                .attr("stroke-width", "0px");
            rowSelection.append("text")
                .attr("transform", "translate(10,15)");
                //on("mouseover",fMouseOver);
        };

        var rowUpdate = function(rowSelection) {

            rowSelection.select("rect")
                .attr("fill", function(d,i) { console.log(d.label);
                    if (i%2==0) {return '#FFFFFF'} else return '#E7E7E7'});//change id to label
                //})
                //.on("click",fMouseClick);
            rowSelection.select("text")
                .text(function (d,i) {
                  if(i==0){
                    return (d.date) + '\xa0\xa0\xa0\xa0\xa0\xa0\xa0' + d.sentimentClass + '\xa0\xa0\xa0\xa0' +  d.urgency + '\xa0\xa0\xa0\xa0\xa0\xa0' + d.label;
                  }else{
                    return (d.date) + '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0' 
                        + d.sentimentClass + '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0' +  
                        d.urgency + '\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0' + d.label;
                  }
                })
                .attr('fill',function(d, i) {
                  if(i == 0){
                    return 'black';
                  }else{
                    return colorScale(d.colour)
                  }})
                .attr('font-size',function(d, i) {
                  if(i == 0){
                    return 14;
                  }else{
                    return 12
                  }})
                .attr('font-weight', function(d, i) {
                  if(i == 0){
                    return 600;
                  }else{
                    return 400;
                  }});
        };

        var rowExit = function(rowSelection) {
        };
         

        var virtualScroller = d3.VirtualScroller()
            .rowHeight(30)
            .enter(rowEnter)
            .update(rowUpdate)
            .exit(rowExit)
            .svg(scrollSVG1)
            .totalRows(maxim+1)
            .viewport(d3.select(".viewport1"));

        // tack on index to each data item for easy to read display
       

//console.log(states.length) 
console.log(maxim) 


        virtualScroller.data(states, function(d) { return d.label; });//change id to label

        chartGroup.call(virtualScroller);

        function createFilters(svgDefs) {
            var filter = svgDefs.append("svg:filter")
                .attr("id", "dropShadow1")
                .attr("x", "0")
                .attr("y", "0")
                .attr("width", "200%")
                .attr("height", "200%");

            filter.append("svg:feOffset")
                .attr("result", "offOut")
                .attr("in", "SourceAlpha")
                .attr("dx", "1")
                .attr("dy", "1");

            filter.append("svg:feColorMatrix")
                .attr("result", "matrixOut")
                .attr("in", "offOut")
                .attr("type", "matrix")
                .attr("values", "0.1 0 0 0 0 0 0.1 0 0 0 0 0 0.1 0 0 0 0 0 0.2 0");

            filter.append("svg:feGaussianBlur")
                .attr("result", "blurOut")
                .attr("in", "matrixOut")
                .attr("stdDeviation", "1");

            filter.append("svg:feBlend")
                .attr("in", "SourceGraphic")
                .attr("in2", "blurOut")
                .attr("mode", "normal");
        }

})//d3.csv for ffmouseclick ends
}//ffmouseclick ends
// d3.csv ends






  })// end Data

  function type(d) {
    //console.log(d.Close);
    return {

      date    : parseDate(d.Date),
      price   : +d.Close,
      average : +d.Average,
      volume : +d.Volume,
      Expected:+d.Expected,
      Predicted:+d.Predicted,
      Open:+d.Open
    }
  }
}

//fMouseClick()

//Creation of Virtual Scroll bar containing Asset Names
    //urln='/viewdb'
    //url_for('static', filename='asset.json')
   



}); // d3.json ends
}());// main function ends