//ndarray FULL modern refactor

function order() {
  let stride = this.stride
  let terms = new Array(stride.length)
  let i
  for(i=0; i<terms.length; i++) {
    terms[i] = [Math.abs(stride[i]), i]
  }
  terms.sort((a,b)=>a[0]-b[0])
  let result = new Array(terms.length)
  for(i=0; i<result.length; i++) {
    result[i] = terms[i][1]
  }
  return result
}

function compileConstructor(dtype, dimension) {
  let className = ["View", dimension, "d", dtype].join("")
  if(dimension < 0) {className = "View_Nil" + dtype}
  let useGetters = (dtype === "generic")
  
  let dimIs = Math.min(dimension+1,2)

  let indices = Array(Math.max(dimension,0)).fill(1).map((_,i)=>i)
    
  procedure = function(){ 
    let CTOR_LIST = CACHED_CONSTRUCTORS[dtype], TrivialArray = CTOR_LIST[0]
    let indexAt = (args,bind)=>{return bind.offset+indices.reduce((sum,cur)=>sum+bind.stride[cur]*args[cur],0)}
    let fn = {
      [className](data,shape,stride,offset){
        this.data = data
        this.shape = dimIs==2?shape:this.shape
        this.stride = dimIs==2?stride:this.stride
        this.offset = [void 0,shape,offset][dimIs]
      },
      [className+"_size"]:[()=>0,()=>1,function(){
        return this.shape.reduce((sum,cur)=>sum*cur,1)
      }][dimIs],
      [className+"_set"]:(
        useGetters?
          [()=>{},function(v){return this.data.set(this.offset,v)},function(...args){
            return this.data.set(indexAt(args,this), args[dimension])
          }][dimIs]:
          [()=>{},function(v){return this.data[this.offset]=v},function(...args){
            return this.data[indexAt(args,this)] = args[dimension]
          }][dimIs]
      ),
      [className+"_get"]:(
        useGetters?
          [()=>{},function(){return this.data.get(this.offset)},function(...args){
            return this.data.get(indexAt(args,this))
          }][dimIs]:
          [()=>{},function(){return this.data[this.offset]},function(...args){
            return this.data[indexAt(args,this)]
          }][dimIs]
      ),
      [className+"_index"]:[()=>-1,function(){return this.offset},function(...args){return indexAt(args,this)}][dimIs],
      [className+"_hi"](...args){
        let newShape = indices.map((e)=>(typeof args[e]!=='number'||args[e]<0)?this.shape[e]:args[e]|0)
        return new fn[className](this.data, newShape, this.stride, this.offset)
      },
      [className+"_lo"](...args){
        let b=this.offset,c=[...this.shape],d=0

        indices.map((e)=>{
          if(typeof args[e]==='number'&&args[e]>=0){
            d=args[e]|0;
            b+=this.stride[e]*d;
            c[e]-=d
          }
        })
    
        return new fn[className](this.data, c, this.stride, b)
      },
      [className+"_step"](...args){
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
          
        return new fn[className](this.data, a, b, c)
      },
      [className+"_transpose"](...args){
        args = args.map(function(n,idx) {return n=((n===void 0)?idx:n|0)})
        
        let tShape = [...indices].map((e)=>this.shape[args[e]])
        let tStride = [...indices].map((e)=>this.stride[args[e]])
        
        return new fn[className](this.data, tShape, tStride, this.offset)
      },
      [className+"_pick"]:[()=>null,function(){return TrivialArray(this.data)},function(...args){
        let a=[],b=[],c=this.offset
      
        indices.map((e)=>{
          if(typeof args[e]==='number'&&args[e]>=0){
            c=(c+this.stride[e]*args[e])|0
          }else{
            a.push(this.shape[e]);b.push(this.stride[e])
          }
        })
        
        return CTOR_LIST[a.length+1](this.data,a,b,c)
      }][dimIs],
      [className+"_copy"]:[
        function(){return new fn[className](this.data)},
        function(){return new fn[className](this.data,this.offset)},
        void 0
      ][dimIs],
      ["construct_"+className](data,shape,stride,offset){
        return new fn[className](data, shape, stride, offset)
      }
    }
    
    let proto=fn[className].prototype;
    proto.valueOf=!dimension?fn[className+"_get"]:proto.valueOf
    proto.dtype=dtype;
    proto.dimension=dimension;
    Object.defineProperty(proto,'size',{get:fn[className+"_size"]})
    Object.defineProperty(proto,'order',{get:order,configurable:true})
    proto.set=fn[className+"_set"]
    proto.get=fn[className+"_get"]
    proto.index=fn[className+"_index"]
    proto.hi=fn[className+"_hi"]
    proto.lo=fn[className+"_lo"]
    proto.step=fn[className+"_step"]
    proto.transpose=fn[className+"_transpose"]
    proto.pick=fn[className+"_pick"]
    
    if(dimIs!==2){
      proto.shape=proto.stride=proto.order=[];
      proto.lo=proto.hi=proto.transpose=proto.step=fn[className+"_copy"]
    }
    
    return fn["construct_"+className]
  }

  return procedure()
}

function arrayDType(data) {
  if(data?.constructor?.isBuffer?.(data)) {
    return "buffer"
  }
  if(typeof Float64Array !== ""+void 0) {
    const type = Object.prototype.toString.call(data)
      .slice(8, -1)
      .replace("Clamped", "_clamped")
      .replace(/(?<=\w)Array/g, "")
      .toLowerCase()
    if(type in CACHED_CONSTRUCTORS) return type
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
  if(data === void 0) {
    let ctor = CACHED_CONSTRUCTORS.array[0]
    return ctor([])
  } else if(typeof data === "number") {
    data = [data]
  }
  if(shape === void 0) {
    shape = [ data.length ]
  }
  let d = shape.length
  if(stride === void 0) {
    stride = new Array(d)
    for(let i=d-1, sz=1; i>=0; i--) {
      stride[i] = sz
      sz *= shape[i]
    }
  }
  if(offset === void 0) {
    offset = 0
    for(let i=0; i<d; i++) {
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


var mat = ndarray(new Float64Array([1, 0, 0, 1]), [2,2])

console.log(mat)
//Now:
//
// mat = 1 0
//       0 1
//
