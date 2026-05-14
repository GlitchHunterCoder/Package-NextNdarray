//ndarray FULL modern refactor

function order() {
  let stride = this.stride
  let terms = new Array(stride.length)
  let i
  for(i=0; i<terms.length; ++i) {
    terms[i] = [Math.abs(stride[i]), i]
  }
  terms.sort((a,b)=>a[0]-b[0])
  let result = new Array(terms.length)
  for(i=0; i<result.length; ++i) {
    result[i] = terms[i][1]
  }
  return result
}

function compileConstructor(dtype, dimension) {
  let className = ["View", dimension, "d", dtype].join("")
  if(dimension < 0) {
    className = "View_Nil" + dtype
  }
  let useGetters = (dtype === "generic")

  if(dimension === -1) {
    //Special case for trivial arrays
  
    let fn = {
      [className]:function(a){this.data=a},
      ["construct_"+className]:function(a){return new fn[className](a)}
    }
    let proto=fn[className].prototype;
    proto.dtype=dtype;
    proto.index=function(){return -1};
    proto.size=0;
    proto.dimension=-1;
    proto.shape=proto.stride=proto.order=[];
    proto.lo=proto.hi=proto.transpose=proto.step=function(){return new fn[className](this.data);};
    proto.get=proto.set=function(){};
    proto.pick=function(){return null};
    return fn["construct_"+className]

  } else if(dimension === 0) {
    //Special case for 0d arrays

    let TrivialArray = CACHED_CONSTRUCTORS[dtype][0]
    let fn = {
      [className]:function(a,d){this.data=a;this.offset=d},
      [className+"_copy"]:function(){return new fn[className](this.data,this.offset)},
      [className+"_pick"]:function(){return TrivialArray(this.data)},
      [className+"_get"]:useGetters ? function(){return this.data.get(this.offset)} : function(){return this.data[this.offset]},
      [className+"_set"]:useGetters ? function(v){return this.data.set(this.offset,v)} : function(v){return this.data[this.offset]=v},
      ["construct_"+className]:function(a,b,c,d){return new fn[className](a,d)}
    }
    
    let proto=fn[className].prototype;
    proto.dtype=dtype;
    proto.index=function(){return this.offset};
    proto.dimension=0;
    proto.size=1;
    proto.shape=proto.stride=proto.order=[];
    proto.lo=proto.hi=proto.transpose=proto.step=fn[className+"_copy"]
    proto.pick=fn[className+"_pick"]
    proto.valueOf=proto.get=fn[className+"_get"]
    proto.set=fn[className+"_set"]
    return fn["construct_"+className]
  }

  //easy cached iterated over array
  let indices = Array(dimension).fill(1).map((_,i)=>i)
    
  procedure = function(){
        
    let CTOR_LIST = CACHED_CONSTRUCTORS[dtype], ORDER = order
    
    let fn = {
      [className]: function(a, ...args) {
        this.data = a
        this.shape = args.slice(0, dimension)
        this.stride = args.slice(dimension, dimension*2)
        this.offset = args[dimension*2] | 0
      },
      [className+"_size"]:function(){
        return indices.map((e)=>this.shape[e]).reduce((sum,cur)=>sum*cur,1)
      },
      [className+"_set"]:useGetters?function(...args){
        return this.data.set(this.offset+indices.reduce((sum,cur)=>sum+this.stride[cur]*args[cur],0), args[dimension])
      }:function(...args){
        return this.data[this.offset+indices.reduce((sum,cur)=>sum+this.stride[cur]*args[cur],0)] = args[dimension]
      },
      [className+"_get"]:useGetters?function(...args){
        return this.data.get(this.offset+indices.reduce((sum,cur)=>sum+this.stride[cur]*args[cur],0))
      }:function(...args){
        return this.data[this.offset+indices.reduce((sum,cur)=>sum+this.stride[cur]*args[cur],0)]
      },
      [className+"_index"]:function(...args){
        return this.offset+indices.reduce((sum,cur)=>sum+this.stride[cur]*args[cur],0)
      },
      [className+"_hi"]:function(...args){
        return new fn[className](
          this.data,
          ...indices.map((e)=>(typeof args[e]!=='number'||args[e]<0)?this.shape[e]:args[e]|0),
          ...this.stride,
          this.offset
        )
      },
      [className+"_lo"]:function(...args){
        let b=this.offset,c=[...this.shape],d=0

        indices.map((e)=>{
          if(typeof args[e]==='number'&&args[e]>=0){
            d=args[e]|0;
            b+=this.stride[e]*d;
            c[e]-=d
          }
        })
    
        return new fn[className](
            this.data,
            ...c,
            ...this.stride,
            b
        )
      },
      [className+"_step"]:function(...args){
        let a=[...this.shape],b=[...this.stride],c=this.offset,d=0
    
        indices.map((e)=>{
          if(typeof args[e]==='number'){
            d=args[e]|0;
            if(d<0){
              c+=this.stride[e]*(this.shape[e]-1);
              a[e]=Math.ceil(-this.shape[e]/d)
            }else{
              a[e]=Math.ceil(this.shape[e]/d)
            }
            b[e]*=d
          }
        })
          
        return new fn[className](
          this.data,
          ...a,
          ...b,
          c
        )
      },
      [className+"_transpose"]:function(...args){
        args = args.map(function(n,idx) {return n=(n===undefined?idx:n|0)})
        
        let tShape = [...indices].map((e)=>this.shape[args[e]])
        let tStride = [...indices].map((e)=>this.stride[args[e]])
        
        return new fn[className](
          this.data,
          ...tShape,
          ...tStride,
          this.offset
        )
      },
      [className+"_pick"]:function(...args){
        let a=[],b=[],c=this.offset
      
        indices.map((e)=>{
          if(typeof args[e]==='number'&&args[e]>=0){c=(c+this.stride[e]*args[e])|0}else{a.push(this.shape[e]);b.push(this.stride[e])}
        })
        
        return CTOR_LIST[a.length+1](this.data,a,b,c)
        
      },
      ["construct_"+className]:function(data,shape,stride,offset){
        return new fn[className](data,
          ...shape,
          ...stride,
          offset
        )
      }
    }
    
    let proto=fn[className].prototype;
    proto.dtype=dtype;
    proto.dimension=dimension;
    Object.defineProperty(proto,'size',{get:fn[className+"_size"]})
    Object.defineProperty(proto,'order',{get:ORDER})
    proto.set=fn[className+"_set"]
    proto.get=fn[className+"_get"]
    proto.index=fn[className+"_index"]
    proto.hi=fn[className+"_hi"]
    proto.lo=fn[className+"_lo"]
    proto.step=fn[className+"_step"]
    proto.transpose=fn[className+"_transpose"]
    proto.pick=fn[className+"_pick"]
    return fn["construct_"+className]
  }

  return procedure()
}

function arrayDType(data) {
  if(data?.constructor?.isBuffer?.(data)) {
    return "buffer"
  }
  if((typeof Float64Array) !== "undefined") {
    switch(Object.prototype.toString.call(data)) {
      case "[object Float64Array]":
        return "float64"
      case "[object Float32Array]":
        return "float32"
      case "[object Int8Array]":
        return "int8"
      case "[object Int16Array]":
        return "int16"
      case "[object Int32Array]":
        return "int32"
      case "[object Uint8Array]":
        return "uint8"
      case "[object Uint16Array]":
        return "uint16"
      case "[object Uint32Array]":
        return "uint32"
      case "[object Uint8ClampedArray]":
        return "uint8_clamped"
      case "[object BigInt64Array]":
        return "bigint64"
      case "[object BigUint64Array]":
        return "biguint64"
    }
  }
  if(Array.isArray(data)) {
    return "array"
  }
  return "generic"
}

let CACHED_CONSTRUCTORS = {
  "float32":[],
  "float64":[],
  "int8":[],
  "int16":[],
  "int32":[],
  "uint8":[],
  "uint16":[],
  "uint32":[],
  "array":[],
  "uint8_clamped":[],
  "bigint64": [],
  "biguint64": [],
  "buffer":[],
  "generic":[]
}

/*;(function() {
  for(let id in CACHED_CONSTRUCTORS) {
    CACHED_CONSTRUCTORS[id].push(compileConstructor(id, -1))
  }
});
//use if you want precompiled caches
*/

function wrappedNDArrayCtor(data, shape, stride, offset) {
  if(data === undefined) {
    let ctor = CACHED_CONSTRUCTORS.array[0]
    return ctor([])
  } else if(typeof data === "number") {
    data = [data]
  }
  if(shape === undefined) {
    shape = [ data.length ]
  }
  let d = shape.length
  if(stride === undefined) {
    stride = new Array(d)
    for(let i=d-1, sz=1; i>=0; --i) {
      stride[i] = sz
      sz *= shape[i]
    }
  }
  if(offset === undefined) {
    offset = 0
    for(let i=0; i<d; ++i) {
      if(stride[i] < 0) {
        offset -= (shape[i]-1)*stride[i]
      }
    }
  }
  let dtype = arrayDType(data)
  let ctor_list = CACHED_CONSTRUCTORS[dtype]
  while(ctor_list.length <= d+1) {
    ctor_list.push(compileConstructor(dtype, ctor_list.length-1))
  }
  let ctor = ctor_list[d+1]
  return ctor(data, shape, stride, offset)
}

let ndarray = wrappedNDArrayCtor
